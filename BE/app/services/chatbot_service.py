import google.generativeai as genai
import pandas as pd
import os
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import uuid  # For generating unique filenames
from typing import Optional, Tuple, Any, Dict, List, Union # For type hinting
import logging # For better internal logging
import sys # For stdout redirection (used by execute_code)
import ast # For code syntax validation
import scipy.stats # Make scipy available for generated code
import webbrowser # To open plots
import platform # Potentially for OS-specific features (imported but not heavily used)
import time # For retry delay
from PIL import Image # For plot analysis
from google.ai.generativelanguage_v1beta.types.content import Content
from google.ai.generativelanguage_v1beta.types.content import Part

# --- Configuration Constants (can be overridden in __init__) ---
DEFAULT_CSV_FILE_PATH = 'supermarket_sales.csv'
DEFAULT_MODEL_NAME = 'gemini-2.5-flash-preview-04-17' # Updated model name
DEFAULT_SAVE_PLOTS_DIR = '../FE/automation-data-analysts/public/plots'
DEFAULT_MAX_UNIQUE_VALUES_FOR_CONTEXT = 20
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_PLOT_FILENAME_PLACEHOLDER = 'generated_plot.png' # Placeholder AI uses
DEFAULT_MAX_HISTORY_TURNS = 10
DEFAULT_MAX_AUTO_FIX_ATTEMPTS = 2
DEFAULT_AUTO_FIX_RETRY_DELAY = 1
DEFAULT_MAX_JOURNEY_ITEMS = 10
DEFAULT_NUM_STARTER_QUESTIONS = 3
DEFAULT_MAX_ROWS_FOR_FULL_CONTEXT_STATS = 50000
DEFAULT_SAMPLE_SIZE_FOR_STATS = 25000

class ChatbotService:
    def __init__(self,
                 csv_file_path: str,
                 model_name: str = DEFAULT_MODEL_NAME,
                 save_plots_dir: str = DEFAULT_SAVE_PLOTS_DIR,
                 api_key: Optional[str] = None,
                 log_level: str = DEFAULT_LOG_LEVEL,
                 max_unique_for_context: int = DEFAULT_MAX_UNIQUE_VALUES_FOR_CONTEXT,
                 max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS,
                 max_auto_fix_attempts: int = DEFAULT_MAX_AUTO_FIX_ATTEMPTS,
                 auto_fix_retry_delay: int = DEFAULT_AUTO_FIX_RETRY_DELAY,
                 max_journey_items: int = DEFAULT_MAX_JOURNEY_ITEMS,
                 num_starter_questions: int = DEFAULT_NUM_STARTER_QUESTIONS,
                 max_rows_for_full_context_stats: int = DEFAULT_MAX_ROWS_FOR_FULL_CONTEXT_STATS,
                 sample_size_for_stats: int = DEFAULT_SAMPLE_SIZE_FOR_STATS,
                 # --- Parameters for hydrating state ---
                 initial_chat_history: Optional[List[Dict[str, Any]]] = None,
                 initial_journey_log: Optional[List[Dict[str, Any]]] = None,
                 initial_focus_filter: Optional[str] = None,
                 initial_pending_code: Optional[Dict[str, Any]] = None,
                 initial_pending_whatif: Optional[Dict[str, Any]] = None,
                 initial_pending_focus_proposal: Optional[Dict[str, Any]] = None,
                 initial_last_plot_path: Optional[str] = None,
                 initial_auto_execute_enabled: bool = True # Default to True
                 ):

        self.csv_file_path = csv_file_path
        self.model_name = model_name
        self.save_plots_dir = save_plots_dir
        self.max_unique_for_context = max_unique_for_context
        self.max_history_turns = max_history_turns
        self.max_auto_fix_attempts = max_auto_fix_attempts
        self.auto_fix_retry_delay = auto_fix_retry_delay
        self.max_journey_items = max_journey_items
        self.num_starter_questions = num_starter_questions
        self.max_rows_for_full_context_stats = max_rows_for_full_context_stats
        self.sample_size_for_stats = sample_size_for_stats
        self.last_executed_plot_path = initial_last_plot_path

        self._setup_logging(log_level)
        os.makedirs(self.save_plots_dir, exist_ok=True)

        self.api_key = api_key or self._load_api_key()
        if not self.api_key:
            raise ValueError("Google API Key not found or provided.")
        genai.configure(api_key=self.api_key)
        logging.info("Google AI configured.")

        self.df_main = self._load_dataframe(self.csv_file_path)
        if self.df_main is None:
            raise ValueError(f"Failed to load DataFrame from {self.csv_file_path}")

        self.data_context_string = self._get_enhanced_data_context(self.df_main, self.max_unique_for_context)
        
        self.model = genai.GenerativeModel(self.model_name)
        
        if initial_chat_history:
            try:
                self.chat_session = self.model.start_chat(history=initial_chat_history)
                logging.info(f"ChatbotService initialized WITH existing chat history (length {len(initial_chat_history)}) for {os.path.basename(csv_file_path)}.")
            except Exception as e:
                logging.error(f"Error initializing chat with existing history: {e}. Falling back to new chat.", exc_info=True)
                self.chat_session = self._initialize_chat_session() # Fallback
        else:
            self.chat_session = self._initialize_chat_session()
            logging.info(f"ChatbotService initialized with NEW chat session for {os.path.basename(csv_file_path)}.")
        
        # Corrected State variable initialization:
        self.current_focus_filter: Optional[str] = initial_focus_filter
        self.analysis_journey_log: List[Dict[str, Any]] = initial_journey_log if initial_journey_log is not None else []
        self.auto_execute_enabled: bool = initial_auto_execute_enabled # Use the passed parameter

        self.pending_code_to_execute: Optional[Dict[str, Any]] = initial_pending_code
        self.pending_whatif_code_to_execute: Optional[Dict[str, Any]] = initial_pending_whatif
        self.pending_focus_proposal: Optional[Dict[str, Any]] = initial_pending_focus_proposal
        self.last_executed_plot_path: Optional[str] = initial_last_plot_path
        
    def _serialize_part(self, part: Part) -> Dict[str, Any]:
        part_dict = {}
        if hasattr(part, 'text') and part.text: # Check if text attribute exists and is not None/empty
            part_dict["text"] = part.text
        
        # Optional: Handle other part types like inline_data, function_call, function_response
        # if hasattr(part, 'inline_data') and part.inline_data:
        #     part_dict["inline_data"] = {
        #         "mime_type": part.inline_data.mime_type,
        #         "data_placeholder": f"<Inline data: {part.inline_data.mime_type}>"
        #     }
        # ... etc. for function_call, function_response
        
        if not part_dict and hasattr(part, 'text'): # If no other content but text was None/empty, still provide text key
             part_dict["text"] = "" # Ensure 'text' key exists even if empty, as API might expect it for text parts

        return part_dict if part_dict else {"text": ""} # Ensure a valid part dict is returned
    
    def _serialize_chat_history_turn(self, turn: Content) -> Dict[str, Any]:
        """Converts a single genai.types.Content object (a turn in history) to a JSON-serializable dict."""
        if not isinstance(turn, Content):
            logging.error(f"SERIALIZATION ERROR: Expected genai.types.Content, got {type(turn)}. Data: {str(turn)[:200]}")
            return {"role": "error_internal", "parts": [{"text": "Serialization error: Invalid turn type."}]}

        # Ensure role is valid, otherwise default or log warning
        valid_roles = ("user", "model", "function") # function role is for function calling
        role = turn.role
        if role not in valid_roles:
            logging.warning(f"Serializing turn with an API-unknown role: '{role}'. Using 'user' as fallback for structure.")
            # This is a choice: either raise error, or try to make it structurally valid.
            # The API only accepts 'user' and 'model' for standard chat history passed to start_chat,
            # unless function calling is involved where 'function' role is also used for responses.
            # For simple history, it must be user/model.
            if role != "tool": # "tool" role is for function responses (formerly "function")
                 role = "user" # A somewhat arbitrary fallback to maintain structure for non-function turns

        serialized_parts = []
        if turn.parts: # Ensure parts list exists
            for p in turn.parts:
                if p: # Ensure part itself is not None
                    serialized_parts.append(self._serialize_part(p))
        
        return {
            "role": role,
            "parts": serialized_parts
        }
        

    def get_persistable_ai_history(self) -> List[Dict[str, Any]]:
        """
        Serializes the current AI chat session history into a list of dictionaries
        suitable for JSON storage and for re-initializing a chat session.
        """
        if self.chat_session and hasattr(self.chat_session, 'history') and self.chat_session.history:
            # self.chat_session.history is List[genai.types.Content]
            return [self._serialize_chat_history_turn(turn) for turn in self.chat_session.history]
        return [] # Return empty list if no history

    def get_current_state_for_persistence(self) -> Dict[str, Any]:
        """Extracts all persistable state from the service instance."""
        return {
            "chat_history": self.get_persistable_ai_history(), # Use the dedicated method
            "analysis_journey_log": self.analysis_journey_log,
            "current_focus_filter": self.current_focus_filter,
            "pending_code_to_execute": self.pending_code_to_execute,
            "pending_whatif_code_to_execute": self.pending_whatif_code_to_execute, # if you have it
            "pending_focus_proposal": self.pending_focus_proposal, # if you have it
            "last_executed_plot_path": self.last_executed_plot_path, # This is the URL
            "auto_execute_enabled": self.auto_execute_enabled
        }
    
    def _setup_logging(self, log_level_str: str):
        log_level_num = getattr(logging, log_level_str.upper(), logging.WARNING)
        # Configure root logger. If this service is part of a larger app,
        # you might want to get a specific logger instance.
        logging.basicConfig(level=log_level_num, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

    def _load_api_key(self) -> Optional[str]:
        load_dotenv()
        try:
            api_key = os.environ['GOOGLE_API_KEY']
            if not api_key:
                logging.error("Error: GOOGLE_API_KEY environment variable is empty.")
                return None
            # Basic format check (optional, can be removed if too restrictive)
            # if not api_key.startswith("AIza"):
            #     logging.warning("Warning: GOOGLE_API_KEY format might be incorrect.")
            return api_key
        except KeyError:
            logging.error("Error: GOOGLE_API_KEY environment variable not found.")
            return None

    def _load_dataframe(self, file_path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Loaded '{file_path}'. Shape: {df.shape}")
            for col in df.select_dtypes(include=['object']).columns:
                # Attempt to convert date-like columns
                if 'date' in col.lower() or 'time' in col.lower() or \
                   (df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').all() and not df[col].isnull().all()): # check if all non-null match pattern
                    try:
                        original_dtype = df[col].dtype
                        converted_col = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                        # Only keep conversion if it's not all NaT and some values were successfully converted
                        if not converted_col.isnull().all() and (df[col].isnull().sum() < len(df[col])):
                            df[col] = converted_col
                            logging.info(f"Successfully converted column '{col}' to datetime.")
                        else: # Revert if conversion failed broadly or made everything NaT
                            # df[col] = pd.read_csv(file_path, usecols=[col]).squeeze("columns") # This might be slow
                            logging.info(f"Reverted conversion for column '{col}' as it resulted in all NaT or was already mostly null.")
                    except Exception as e:
                        logging.warning(f"Could not convert column '{col}' to datetime: {e}", exc_info=False)
            logging.info(f"DataFrame '{os.path.basename(file_path)}' loaded and preprocessed.")
            return df
        except FileNotFoundError:
            logging.error(f"Error: File not found at path '{file_path}'.")
            return None
        except Exception as e:
            logging.exception(f"Error loading or preprocessing CSV from '{file_path}': {e}")
            return None

    def _get_enhanced_data_context(self, dataframe: pd.DataFrame, max_unique_for_context: int) -> str:
        # This function is largely the same as provided in the original script
        context = []
        df_to_describe = dataframe
        sampled_info = ""

        if len(dataframe) > self.max_rows_for_full_context_stats:
            sampled_info = (f"\nNOTE: Due to large dataset size ({len(dataframe)} rows), "
                            f"descriptive statistics and value counts below are based on a "
                            f"random sample of {self.sample_size_for_stats} rows. "
                            f"Overall shape and column info are for the full dataset.")
            context.append(sampled_info)
            df_to_describe = dataframe.sample(n=min(len(dataframe), self.sample_size_for_stats), random_state=42)


        context.append(f"Dataset Shape: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns.")
        context.append("\n--- Column Details (Name, Type, Non-Null Count) ---") # For full dataset
        buffer = io.StringIO()
        # Ensure pandas options are set and reset for the info string
        original_max_rows = pd.options.display.max_rows
        pd.options.display.max_rows = dataframe.shape[1] + 5 # Ensure all columns are shown
        dataframe.info(buf=buffer)
        pd.options.display.max_rows = original_max_rows # Reset
        info_str = buffer.getvalue()
        context.append(info_str)

        missing_values = df_to_describe.isnull().sum() # Use df_to_describe for stats
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            context.append("\n--- Missing Value Counts (from sample if large) ---\n" + missing_values.to_string())
        else:
            context.append("\n--- Missing Value Counts (from sample if large): None ---")

        try:
            numeric_desc = df_to_describe.describe(include=[float, int])
            if not numeric_desc.empty:
                context.append("\n--- Numerical Column Statistics (from sample if large) ---\n" + numeric_desc.to_string())
            else:
                context.append("\n--- Numerical Column Statistics (from sample if large): None ---")
        except Exception as e:
            logging.warning(f"Numeric describe error: {e}")
            context.append("\n--- Numerical Column Statistics (from sample if large): Error generating ---")
        
        try:
            # Include datetime if present, handle bool explicitly if needed
            obj_desc_include_types = ['object', 'category', 'datetime64[ns]', 'bool']
            # Filter types actually present in df_to_describe to avoid errors with describe
            present_types = [t for t in obj_desc_include_types if any(df_to_describe.dtypes == t)]
            if present_types:
                 obj_desc = df_to_describe.describe(include=present_types)
                 if not obj_desc.empty:
                    context.append("\n--- Object/Categorical/Datetime/Boolean Column Statistics (from sample if large) ---\n" + obj_desc.to_string())
                 else: # obj_desc can be empty if all included types have no columns or describe yields nothing
                    context.append("\n--- Object/Categorical/Datetime/Boolean Column Statistics (from sample if large): No descriptive columns or no stats to show ---")
            else:
                 context.append("\n--- Object/Categorical/Datetime/Boolean Column Statistics (from sample if large): No columns of these types in sample ---")

        except Exception as e:
            logging.warning(f"Object/Categorical describe error: {e}")
            context.append("\n--- Object/Categorical/Datetime/Boolean Column Statistics (from sample if large): Error generating ---")


        context.append(f"\n--- Categorical/Object Column Value Counts (Sample, Max {max_unique_for_context} Unique Values Displayed, from sample if large) ---")
        # Use select_dtypes on df_to_describe for value counts
        cat_like_cols = df_to_describe.select_dtypes(include=['object', 'category']).columns
        limited_cats_shown = False
        if not len(cat_like_cols):
            context.append("No string/categorical columns found in the (sampled) data to display value counts for.")
        else:
            for col in cat_like_cols:
                unique_count = df_to_describe[col].nunique()
                if unique_count == 0 and df_to_describe[col].isnull().all(): # All null column
                    context.append(f"\n'{col}': All values are null in the (sampled) data.")
                elif unique_count <= max_unique_for_context:
                    context.append(f"\n'{col}' ({unique_count} Unique Values):\n{df_to_describe[col].value_counts(dropna=False).to_string()}")
                else:
                    context.append(f"\n'{col}': {unique_count} unique values (counts not shown as it exceeds display limit of {max_unique_for_context}).")
                    limited_cats_shown = True
            if limited_cats_shown:
                context.append(f"\n(Note: For columns with more than {max_unique_for_context} unique values, detailed counts are not shown.)")

        context.append("\n--- First 5 Rows (from full dataset) ---\n" + dataframe.head().to_string())
        return "\n".join(context)


    def _initialize_chat_session(self) -> genai.ChatSession:
        system_prompt = f"""You are an expert Data Analyst assistant...
Your primary role is to generate Python code for data analysis, visualization, and "what if" scenarios using a pandas DataFrame `df`.
You also analyze plots, provide insights, assist with filters, and suggest next steps based on the "Analysis Journey".

How This Works:
- DataFrame: The primary DataFrame for queries is `df`. This `df` might be a filtered view of the original data if a focus is active. The original data context summary provided to you indicates if statistics were sampled due to large dataset size.
- Filter Assistance (`/focus` command): Multi-stage process (impact code gen, then AI review with impact).
- "What If" Scenario Explorer (`/whatif` command):
  - User asks a "what if" question.
  - Stage 1 (Code Generation): You generate Python code that creates `df_copy = df.copy()`, applies modifications, and prints the impact. Only Python code block.
  - Stage 2 (Explanation): After code runs, you explain the result.
- General Code Generation: For plots/calculations, operate on `df`. Respect active filters.
- Python environment: `pandas` (as `pd`), `matplotlib.pyplot` (as `plt`), `seaborn` (as `sns`), `scipy.stats` (as `stats`) are available.
- Code Output: ```python ... ```.
  - For calculations: Print result.
  - For visualizations: Save the plot to a file (e.g., `plt.savefig('plot.png')`). CRITICAL: Always call `plt.close('all')` after saving to free memory. Include necessary imports.
    **IMPORTANT for PLOTTING LARGE DATA (e.g., if context indicates >{self.max_rows_for_full_context_stats} rows or stats are sampled):**
    - When asked to plot data that could involve many individual data points (like scatter plots or detailed line plots):
        - Acknowledge that plotting all points directly from a very large dataset can be slow and unreadable.
        - **Prioritize generating code for:**
            1. **Aggregation-based plots:** Histograms (`plt.hist`, `sns.histplot`), Kernel Density Estimates (`sns.kdeplot` for 1D), box plots (`sns.boxplot`), bar plots (of pre-aggregated data like `df.groupby(...).mean()`), heatmaps (for 2D categorical counts or correlations).
            2. **Density plots for 2D relationships:** Hexbin plots (`plt.hexbin` or `sns.jointplot(kind='hex')`), 2D KDE plots (`sns.kdeplot(x="col_x", y="col_y")`).
            3. **Sampled plots:** If a scatter-like view is essential for understanding relationships between two continuous variables, generate code to plot a random sample of the data (e.g., `sampled_df = df.sample(n=min(len(df), 10000), random_state=42)` then plot `sampled_df`).
            4. **Time Series Aggregation:** For time series data with many points, suggest or perform resampling to a coarser granularity (e.g., daily/weekly/monthly averages/sums using `df.resample(...)`) before plotting lines.
        - You can briefly explain *why* you chose a particular plot type (e.g., "To visualize the relationship between X and Y from this large dataset without overplotting, I'll generate a hexbin plot which shows density.").
        - If the user explicitly asks for a type of plot that would be problematic (e.g., "scatter plot all 1 million points"), you should still try to use one of the above strategies (e.g., provide a sampled scatter plot or a density plot) and explain your choice.

Data Context Summary (Original DataFrame - this is also provided to you when you are asked to generate starter questions):
{self.data_context_string}

Your Tasks:
1. Direct Answers (from Data Context).
2. Generate Calculation Code.
3. Generate Visualization Code (respecting large data plotting strategies).
4. Generate Advanced Analysis Code.
5. Assist with Filter Application (`/focus`).
6. Handle "What If" Scenarios (`/whatif` - code gen then explanation).
7. Proactive Insights (Post-Code).
8. Explain Code (Briefly).
9. Acknowledge Limitations.
10. Analyze Plot Images.
11. Suggest Next Steps (based on journey).
12. (Initial Task - Not for user queries) Generate Starter Questions: When first initialized with the data context, you will be asked to provide a few sample questions a non-technical user might ask about the dataset.

IMPORTANT: Ensure the entire output is a single HTML block.
"""
        chat = self.model.start_chat(history=[
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["Understood. I will assist with data analysis, including 'what if' scenarios, filters, and plots, paying special attention to generating appropriate visualizations for large datasets by using aggregation, density plots, or sampling. I will also suggest next steps and can provide starter questions. I will always assume the current DataFrame to work on is `df`."]}
        ])
        return chat

    @staticmethod
    def _extract_python_code(text: str) -> Optional[str]:
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _execute_generated_code(self, code: str, dataframe: pd.DataFrame, user_query: str ="<User Query Not Provided>") -> Tuple[bool, Optional[str], Optional[str]]:
        is_plotting_code = "plt." in code or "sns." in code or "matplotlib" in code or "seaborn" in code
        output_capture = io.StringIO()
        original_stdout = sys.stdout
        raw_save_path = None
        processed_code = code

        try:
            ast.parse(processed_code)
            logging.info("Code syntax validation successful.")
        except SyntaxError as syn_err:
            error_msg = f"Syntax Error: {syn_err}"
            logging.error(f"Generated code failed syntax validation: {error_msg} for query '{user_query}'")
            return False, None, error_msg

        if is_plotting_code:
            unique_id = uuid.uuid4()
            base_filename = f"plot_{user_query[:20].replace(' ','_')}_{unique_id}.png" # Add part of query to filename
            base_filename = re.sub(r'[^\w_.-]', '', base_filename) # Sanitize filename
            raw_save_path = os.path.join(self.save_plots_dir, base_filename)
            save_path_for_code = raw_save_path.replace('\\', '/') # For code injection

            if f"plt.savefig" not in processed_code:
                 # If plt.show() is present, insert savefig before it, otherwise append
                if "plt.show()" in processed_code:
                    processed_code = processed_code.replace("plt.show()", f"plt.savefig('{save_path_for_code}', bbox_inches='tight')\nplt.show()")
                else: # Append savefig if no show and no savefig
                    processed_code += f"\nplt.savefig('{save_path_for_code}', bbox_inches='tight')"
            else: # If savefig is already there, ensure it uses our path
                processed_code = re.sub(r"plt\.savefig\s*\(\s*['\"].*?['\"]\s*", f"plt.savefig('{save_path_for_code}', bbox_inches='tight'", processed_code, count=1)


            if "plt.close('all')" not in processed_code:
                 processed_code += "\nplt.close('all') # Ensure plots are closed to free memory"
            logging.info(f"Executing plotting code for query '{user_query}', targeting save path: {raw_save_path}")
        else:
            logging.info(f"Executing calculation code for query '{user_query}'.")
            sys.stdout = output_capture

        logging.debug(f"--- Code to Execute for query '{user_query}' ---\n{processed_code}\n-----------------------")

        try:
            # Ensure df is available. The AI is instructed to use 'df'.
            # `dataframe` passed to this function is the one to be named 'df' in exec_scope
            exec_scope = {'df': dataframe, 'pd': pd, 'plt': plt, 'sns': sns, 'stats': scipy.stats}
            exec(processed_code, exec_scope)
            
            if not is_plotting_code: # For calculation code
                sys.stdout = original_stdout # Restore stdout before getting value
            captured_output = output_capture.getvalue().strip()

            if is_plotting_code:
                if raw_save_path and os.path.exists(raw_save_path) and os.path.getsize(raw_save_path) > 0:
                    logging.info(f"Plot successfully generated and saved as '{raw_save_path}' for query '{user_query}'.")
                    # In a service, directly opening a browser is often a side effect handled by the client.
                    # For this refactoring, we keep the original behavior.
                    try:
                        webbrowser.open(f'file://{os.path.realpath(raw_save_path)}')
                        logging.info(f"Attempted to open plot file: {raw_save_path}")
                    except Exception as open_err:
                        logging.warning(f"Could not automatically open plot file '{raw_save_path}': {open_err}")
                    return True, raw_save_path, None
                else:
                    logging.warning(f"Plotting code ran for query '{user_query}' but plot file '{raw_save_path}' was not created or is empty.")
                    # Return captured_output which might contain errors or info from the plot script itself
                    return True, captured_output if captured_output else "(Plot file not created or empty)", None 
            else: # Calculation code
                logging.info(f"Calculation code executed successfully for query '{user_query}'. Output captured.")
                return True, captured_output if captured_output else "(No output printed by the code)", None

        except Exception as e:
            if not is_plotting_code:
                sys.stdout = original_stdout # Restore stdout in case of error during exec
            error_msg = f"{type(e).__name__}: {e}"
            logging.error(f"Error executing generated code for query '{user_query}': {error_msg}", exc_info=True) # Log full traceback
            # Clean up plot file if error occurred during plotting
            if is_plotting_code and raw_save_path and os.path.exists(raw_save_path):
                try:
                    os.remove(raw_save_path)
                    logging.info(f"Removed potentially incomplete plot file '{raw_save_path}' due to execution error.")
                except Exception as rm_err:
                    logging.warning(f"Could not remove incomplete plot file '{raw_save_path}': {rm_err}")
            return False, None, error_msg
        finally:
            if sys.stdout != original_stdout : # Ensure stdout is always restored
                sys.stdout = original_stdout
            output_capture.close()
            if is_plotting_code: # Ensure plots are closed even if plt.close('all') was missed in code
                plt.close('all')
    
    def _get_ai_summary_of_finding(self, finding_text: str, original_query: str) -> Optional[str]:
        if not finding_text or len(finding_text) < 20:
            return finding_text

        summary_prompt = f"""The user's query was: "{original_query}"
The result/insight obtained was:
"{finding_text}"

Please provide a very concise one-sentence summary of this key finding. This summary will be used to track the analysis journey.
For example, if the insight was 'The plot shows a positive correlation between X and Y', a good summary would be 'Positive correlation found between X and Y.'
If it was a calculation output like 'Average sales: $150.30', the summary could be 'Calculated average sales as $150.30.'

IMPORTANT: Ensure the entire output is a single HTML block.
"""
        try:
            # This uses the main chat session. For a purely utility function, one might use model.generate_content
            response = self.chat_session.send_message(summary_prompt)
            return response.text.strip()
        except Exception as e:
            logging.warning(f"Could not get AI summary for finding (query: '{original_query}'): {e}")
            return finding_text # Fallback

    def _add_to_journey_log(self, event_type: str, description: str, original_query: str, details: Optional[str] = None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "type": event_type,
            "query": original_query,
            "description": description,
            "details": details if details else ""
        }
        self.analysis_journey_log.append(log_entry)
        logging.info(f"Added to journey: {event_type} - {description} (Query: {original_query})")
        if len(self.analysis_journey_log) > self.max_journey_items:
            self.analysis_journey_log = self.analysis_journey_log[-self.max_journey_items:]
            
    def _get_journey_summary_for_ai(self) -> str:
        if not self.analysis_journey_log:
            return "No analysis steps taken yet."
        
        summary_points = []
        for entry in self.analysis_journey_log:
            point = f"- ({entry['type']}): {entry['description']}"
            # Avoid redundancy if query is already well-represented in description
            if entry['query'] and entry['query'].lower() not in entry['description'].lower() and len(entry['query']) < 60 : 
                point += f" (User query: '{entry['query']}')"
            summary_points.append(point)
        
        return "Summary of recent analysis steps:\n" + "\n".join(summary_points)

    def get_starter_questions(self) -> List[str]:
        starter_prompt = f"""I have loaded a dataset. Here is a summary of its structure and contents:
{self.data_context_string}

Based *only* on this summary, please suggest {self.num_starter_questions} simple and insightful questions that a non-technical user might find interesting to ask about this dataset.
These questions should be answerable either directly from the summary you've just seen, or through simple calculations or visualizations.
Frame them as if the user is asking me (the chatbot). For example:
- "What are the different [relevant categorical column names] in the data?"
- "Can you show me the average [relevant numerical column name] by [relevant categorical column name]?"
- "What's the range of [relevant numerical column name]?"
- "Which [categorical item] has the highest total [numerical value]?"

Provide only the list of questions, each on a new line. Do not number them or add any other text.
"""
        try:
            # Use generate_content for one-off non-chat generation
            starter_response = self.model.generate_content(starter_prompt)
            if starter_response.text:
                questions = starter_response.text.strip().split('\n')
                questions = [re.sub(r"^\s*[-*+]?\s*", "", q).strip() for q in questions if q.strip()]
                return questions[:self.num_starter_questions]
            else:
                logging.warning("AI did not provide starter questions.")
                return []
        except Exception as e:
            logging.error(f"Error generating starter questions: {e}")
            return []
            
    def _get_current_df(self) -> pd.DataFrame:
        """Returns the current working DataFrame, applying focus filter if active."""
        if self.current_focus_filter:
            try:
                # Apply filter to a copy to avoid SettingWithCopyWarning on df_main if it's used later
                df_focused = self.df_main.query(self.current_focus_filter).copy()
                logging.info(f"Operating on filtered DataFrame. Focus: {self.current_focus_filter}. Shape: {df_focused.shape}")
                return df_focused
            except Exception as filter_err:
                logging.error(f"Error applying current focus '{self.current_focus_filter}': {filter_err}. Falling back to main DataFrame.")
                # Fallback to a copy of the main DataFrame
                self._add_to_journey_log("FILTER_ERROR", f"Error applying focus: {self.current_focus_filter}. Reverted to full data.", "SYSTEM (Filter Application)")
                self.current_focus_filter = None # Clear invalid filter
                return self.df_main.copy()
        else:
            logging.info(f"Operating on main DataFrame. Shape: {self.df_main.shape}")
            return self.df_main.copy() # Return a copy to prevent modification of self.df_main by executed code

    def _send_to_ai(self, prompt_content: Union[str, List[Union[str, Image.Image]]], original_user_query: str) -> Optional[str]:
        """Sends content to the AI and manages chat history."""
        try:
            # History management (keep system prompt + user/model turn + N recent turns)
            # The first two items are system prompt and initial model ack.
            if len(self.chat_session.history) > 2 + (self.max_history_turns * 2):
                # Keep system prompt, its ack, and the last N turns
                self.chat_session.history = self.chat_session.history[:2] + self.chat_session.history[-(self.max_history_turns * 2):]
                logging.info(f"Chat history truncated to last {self.max_history_turns} turns.")

            logging.info(f"Sending query to AI (related to user query: '{original_user_query}').")
            if isinstance(prompt_content, str):
                 logging.debug(f"AI Prompt (text only, length {len(prompt_content)}): {prompt_content[:500]}...")
            else: # List, likely with image
                 logging.debug(f"AI Prompt (multi-part, first part): {str(prompt_content[0])[:500]}...")


            response = self.chat_session.send_message(prompt_content)
            ai_response_text = response.text.strip()
            logging.info(f"AI response received for query '{original_user_query}' (length {len(ai_response_text)}).")
            logging.debug(f"AI Raw Response:\n{ai_response_text}")
            return ai_response_text
        except Exception as api_err:
            logging.error(f"API Error during chat with AI for query '{original_user_query}': {api_err}", exc_info=True)
            return None
            
    def _handle_execute_code_interaction(self, code_to_execute: str, original_query: str, is_plot_code: bool,
                                         current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Manages the execution of code, including auto-fix loop if enabled.
        This is called when code is ready to be executed (either immediately or after user confirmation).
        """
        responses = []
        auto_fix_attempts_done = 0
        execution_success = False
        final_result_data = None
        last_error_message = "Execution not attempted."
        current_code_for_this_cycle = code_to_execute

        while auto_fix_attempts_done <= self.max_auto_fix_attempts:
            attempt_msg = f"(Attempt {auto_fix_attempts_done + 1}/{self.max_auto_fix_attempts + 1})"
            logging.info(f"Executing code {attempt_msg} for query: {original_query}")

            success, result_data, error_message = self._execute_generated_code(current_code_for_this_cycle, current_df, original_query)
            last_error_message = error_message

            if success:
                execution_success = True
                final_result_data = result_data
                break # Successful execution, exit retry loop
            
            # Code execution failed
            logging.warning(f"Code execution failed {attempt_msg} for query '{original_query}'. Error: {error_message}")
            responses.append({"type": "execution_error", "message": f"Execution Error {attempt_msg}: {error_message}", "code_attempted": current_code_for_this_cycle})

            auto_fix_attempts_done += 1
            if auto_fix_attempts_done > self.max_auto_fix_attempts:
                break # Max retries reached

            # Conditions to stop auto-fixing (e.g. certain error types, or if auto_execute_enabled is false and it's a retry)
            # For simplicity, if auto_execute is false, we won't auto-fix here. The user can ask the AI to fix it.
            if not self.auto_execute_enabled and auto_fix_attempts_done > 0 : # Only first attempt if not auto-executing fixes
                 logging.info("Auto-execute off, stopping auto-fix attempts after first failure.")
                 break
            if any(err_type in (error_message or "") for err_type in ["SyntaxError", "MemoryError"]): # Basic check
                 logging.warning(f"Error type '{error_message}' not suitable for further auto-fix. Stopping retries.")
                 break

            logging.info(f"Attempting AI auto-fix ({auto_fix_attempts_done}/{self.max_auto_fix_attempts}) for query: {original_query}")
            time.sleep(self.auto_fix_retry_delay)
            
            fix_prompt = f"""The previous Python code execution for the user query '{original_query}' (Current DataFrame focus: {self.current_focus_filter if self.current_focus_filter else "None"}) failed.
Failed Code:
```python
{current_code_for_this_cycle}
Error Message:
{error_message}
Please analyze the error and the code, then provide a corrected version of the Python code.
Provide only the corrected Python code block. Do not include any explanations before or after the code block.
If you believe the error is fundamental and cannot be fixed easily (e.g., requesting an impossible operation on the data), you can respond with just the text "CANNOT_FIX".

IMPORTANT: Ensure the entire output is a single HTML block.
"""
            ai_fix_response_text = self._send_to_ai(fix_prompt, original_query + " (fixing code)")

            if not ai_fix_response_text or ai_fix_response_text == "CANNOT_FIX":
                logging.warning(f"AI could not provide a fix or indicated CANNOT_FIX for query: {original_query}")
                responses.append({"type": "info", "message": "AI could not provide a code fix or indicated it's unfixable."})
                break 

            new_fixed_code = self._extract_python_code(ai_fix_response_text)
            if new_fixed_code:
                responses.append({"type": "ai_code_fix_suggestion", "code": new_fixed_code, "attempt_number": auto_fix_attempts_done})
                current_code_for_this_cycle = new_fixed_code
                if not self.auto_execute_enabled: # If auto-exec is off, user needs to confirm execution of fix
                    self.pending_code_to_execute = {"code": new_fixed_code, "original_query": original_query + f" (fix attempt {auto_fix_attempts_done})", "is_plot": is_plot_code}
                    responses.append({"type": "user_confirmation_needed",
                                  "message": f"AI suggested a fix (Attempt {auto_fix_attempts_done}). To execute, use /execute_pending_code.",
                                  "details": "The previous execution failed."})
                    execution_success = False # Mark as not successful yet, pending user action for the fix
                    return {"status": "pending_fix_confirmation", "responses": responses, "pending_code": self.pending_code_to_execute}
            else:
                logging.warning(f"AI response for fix did not contain a valid code block for query: {original_query}")
                responses.append({"type": "info", "message": "AI fix response did not contain a code block."})
                break # AI didn't provide code, stop retrying

        # After retry loop
        if execution_success:
            code_type_log = "PLOT_GENERATED" if is_plot_code else "CALCULATION_PERFORMED"
            desc_log = f"Successfully executed code for: {original_query}."
            details_log = final_result_data

            if is_plot_code and final_result_data and isinstance(final_result_data, str) and os.path.exists(final_result_data):
                self.last_executed_plot_path = final_result_data
                responses.append({"type": "plot_generated", "path": final_result_data, "message": f"Plot saved to: {final_result_data}",
                                "next_actions": ["/analyze_last_plot"]})
                desc_log = f"Generated plot: {os.path.basename(final_result_data)} for query: {original_query}"
            elif not is_plot_code:
                responses.append({"type": "calculation_result", "output": final_result_data or "(No textual output)", 
                                "message": "Calculation code executed."})
                summary = self._get_ai_summary_of_finding(final_result_data or "", original_query)
                desc_log = summary or desc_log
            
            self._add_to_journey_log(code_type_log, desc_log, original_query, details=details_log)
            return {"status": "success", "responses": responses, "result_data": final_result_data}
        else:
            self._add_to_journey_log("CODE_EXECUTION_FAILED", f"Failed to execute code for: {original_query}. Last error: {last_error_message}", original_query, details=f"Last attempted code:\n{current_code_for_this_cycle}\nError:\n{last_error_message}")
            responses.append({"type": "final_execution_failure", "message": f"Code execution failed after {auto_fix_attempts_done} attempt(s). Last error: {last_error_message}", "last_error": last_error_message})
            return {"status": "failure", "responses": responses, "error_message": last_error_message}

    def process_user_query(self, user_query: str, image_attachment: Optional[Image.Image] = None) -> List[Dict[str, Any]]:
        """
        Main method to process user queries.
        Returns a list of response dictionaries.
        Each dictionary has a "type" and other relevant fields.
        """
        responses: List[Dict[str, Any]] = []

        # Get current DataFrame based on focus
        current_df = self._get_current_df()

        original_user_query_for_context = user_query # Keep a pristine copy for logs/prompts

        if user_query.strip().startswith('/'):
            parts = user_query.strip().split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            # --- COMMAND HANDLING ---
            if command == '/quit': # Client should handle this, but service can acknowledge
                responses.append({"type": "info", "message": "Quit command received. Session ending."})
                # No actual quitting here, client controls lifecycle
            elif command == '/help':
                help_text = """Available Commands:
/help - Show this help message.
/config setting [value] - Configure settings (e.g., /config auto_execute on|off).
/context - Show the data summary context sent to the AI.
/starter_questions - Get some initial questions to ask.
/focus [condition] - Get AI advice and impact for a filter.
/apply_focus [yes|no|custom <new_condition>] - Act on a proposed focus.
/clear_focus - Remove the active focus filter.
/whatif [scenario] - Explore a 'what if' scenario (AI generates code).
/execute_pending_whatif_code - Execute AI-generated code for a 'whatif' scenario.
/history_summary - Show current chat history message count (for AI context).
/clear_history - Clear chat history (keeps system prompt).
/journey - Show the recorded analysis journey.
/execute_pending_code - Execute the last AI-generated code if confirmation was needed.
/analyze_last_plot - Ask AI for insights on the most recently generated plot.
/suggest_next_steps - Ask AI for next step suggestions based on current analysis.
"""
                responses.append({"type": "help_text", "content": help_text})
            elif command == '/config':
                if args:
                    config_parts = args.split(maxsplit=1)
                    setting_name = config_parts[0].lower()
                    setting_value = config_parts[1].lower() if len(config_parts) > 1 else None
                    if setting_name == "auto_execute":
                        if setting_value == "on":
                            self.auto_execute_enabled = True
                            responses.append({"type": "config_update", "message": "Auto-execution ENABLED."})
                        elif setting_value == "off":
                            self.auto_execute_enabled = False
                            responses.append({"type": "config_update", "message": "Auto-execution DISABLED."})
                        else:
                            responses.append({"type": "error", "message": f"Invalid value for auto_execute: '{setting_value}'. Use 'on' or 'off'."})
                    else:
                        responses.append({"type": "error", "message": f"Unknown config setting: '{setting_name}'."})
                else:
                    responses.append({"type": "info", "message": f"Current auto_execute mode: {'ON' if self.auto_execute_enabled else 'OFF'}. Usage: /config auto_execute [on|off]"})
            
            elif command == '/context':
                responses.append({"type": "data_context", "content": self.data_context_string})
            
            elif command == '/starter_questions':
                questions = self.get_starter_questions()
                if questions:
                    responses.append({"type": "starter_questions", "questions": questions})
                else:
                    responses.append({"type": "info", "message": "Could not generate starter questions at this time."})
            
            elif command == '/focus':
                if not args:
                    msg = f"Current focus: {self.current_focus_filter}" if self.current_focus_filter else "No focus active."
                    responses.append({"type": "info", "message": f"Usage: /focus [pandas query condition]. {msg}"})
                else:
                    user_filter_condition = args.strip()
                    logging.info(f"Processing /focus command with condition: {user_filter_condition}")
                    
                    # Stage 1: AI generates impact calculation code
                    impact_code_prompt = f"""The user wants to apply the filter condition: `{user_filter_condition}`
The full, original DataFrame is available as df. (The data context summary I have is for this original df).

Your task: Generate Python code to calculate and print() the impact of applying this filter.
The code should print():
    a. The original number of rows in df.
    b. The number of rows remaining if the filter {user_filter_condition} is applied.
    c. Optional but preferred: If a prominent numerical column (e.g., 'Total', 'Sales', 'Gross income', based on the Data Context Summary) exists, also print() the sum/mean of this column for the original df and for the filtered df. Choose ONE such column. Clearly label these print outputs.

Provide ONLY the Python code block. No explanations before or after the code block.

IMPORTANT: Ensure the entire output is a single HTML block.
"""
                    ai_response_impact_code = self._send_to_ai(impact_code_prompt, f"/focus impact code gen for {user_filter_condition}")
                    impact_code = self._extract_python_code(ai_response_impact_code or "")

                    impact_assessment_output = "Impact assessment could not be performed (AI did not generate code)."
                    if impact_code:
                        logging.info(f"Executing impact assessment code for /focus '{user_filter_condition}'")
                        # Execute this code on df_main (as per prompt to AI)
                        success, impact_data, error_msg = self._execute_generated_code(impact_code, self.df_main, f"/focus impact calc for {user_filter_condition}")
                        if success:
                            impact_assessment_output = impact_data if impact_data else "Impact code ran, no direct output printed."
                        else:
                            impact_assessment_output = f"Error executing impact code: {error_msg}"
                            logging.error(f"Impact code execution failed for /focus '{user_filter_condition}': {error_msg}")
                    
                    # Stage 2: AI reviews filter with impact context
                    review_prompt = f"""User's proposed filter condition: `{user_filter_condition}`
                
The calculated impact of applying this filter to the original DataFrame was:
--- BEGIN IMPACT OUTPUT ---
{impact_assessment_output}
--- END IMPACT OUTPUT ---

Data Context Summary (original DataFrame):
{self.data_context_string}

Tasks:

Review the user's filter condition ({user_filter_condition}) in light of its calculated impact and the data context.

Is the syntax likely valid (brief comment)?

Based on the impact and data context, are there any obvious issues or more insightful alternative ways to filter related to the user's likely intent? (e.g., if impact is too drastic, or too minor, or if other categorical values might be relevant).

Keep suggestions concise and directly actionable.

Provide ONLY your textual review and suggestions. Do NOT generate any Python code in this response.

IMPORTANT: Ensure the entire output is a single HTML block.
"""
                    ai_filter_review_text = self._send_to_ai(review_prompt, f"/focus AI review for {user_filter_condition}") or "AI review not available."

                    self.pending_focus_proposal = {
                        "filter_condition": user_filter_condition,
                        "impact_assessment": impact_assessment_output,
                        "ai_review": ai_filter_review_text
                    }
                    responses.append({
                        "type": "focus_proposal_ready",
                        "filter_condition": user_filter_condition,
                        "impact_assessment": impact_assessment_output,
                        "ai_review": ai_filter_review_text,
                        "next_actions": [
                            f"/apply_focus yes", 
                            f"/apply_focus custom <enter your pandas query condition>",
                            f"/apply_focus no"
                        ],
                        "message": "AI has reviewed the filter. Use /apply_focus to proceed."
                    })
                    self._add_to_journey_log("FOCUS_PROPOSED", f"Proposed filter: '{user_filter_condition}'. Review provided.", original_user_query_for_context, 
                    details=f"Impact: {impact_assessment_output}\nReview: {ai_filter_review_text}")


            elif command == '/apply_focus':
                if not self.pending_focus_proposal:
                    responses.append({"type": "error", "message": "No focus proposal is pending. Use /focus first."})
                else:
                    decision = args.split(maxsplit=1)[0].lower() if args else "no" # Default to 'no' if no args
                    original_proposed_filter = self.pending_focus_proposal["filter_condition"]
                    applied_filter_desc = ""
                    
                    if decision == 'yes':
                        try:
                            self.df_main.query(original_proposed_filter) # Test on main df
                            self.current_focus_filter = original_proposed_filter
                            applied_filter_desc = f"Focus set to: {self.current_focus_filter}"
                            responses.append({"type": "info", "message": f"Focus set: {self.current_focus_filter}"})
                        except Exception as q_err:
                            self.current_focus_filter = None # Clear if errored
                            applied_filter_desc = f"Failed to apply proposed filter '{original_proposed_filter}' due to error: {q_err}"
                            responses.append({"type": "error", "message": applied_filter_desc})
                    elif decision == 'custom':
                        custom_filter_condition = args.split(maxsplit=1)[1] if len(args.split(maxsplit=1)) > 1 else ""
                        if custom_filter_condition:
                            try:
                                self.df_main.query(custom_filter_condition) # Test
                                self.current_focus_filter = custom_filter_condition
                                applied_filter_desc = f"Focus set to custom filter: {self.current_focus_filter}"
                                responses.append({"type": "info", "message": f"Focus set to custom filter: {self.current_focus_filter}"})
                            except Exception as q_err:
                                self.current_focus_filter = None
                                applied_filter_desc = f"Failed to apply custom filter '{custom_filter_condition}': {q_err}"
                                responses.append({"type": "error", "message": applied_filter_desc})
                        else:
                            applied_filter_desc = "User chose custom filter but provided none."
                            responses.append({"type": "info", "message": "No custom filter provided. Focus not changed."})
                    else: # 'no' or anything else
                        applied_filter_desc = "User chose not to apply the proposed filter."
                        responses.append({"type": "info", "message": f"Filter not applied. Focus remains: {self.current_focus_filter or 'None'}"})
                    
                    self._add_to_journey_log("FOCUS_APPLIED_OR_REJECTED", applied_filter_desc, f"/apply_focus (original: {original_proposed_filter})", 
                                            details=f"Original Proposal:\n{self.pending_focus_proposal}")
                    self.pending_focus_proposal = None # Clear pending proposal

            elif command == '/clear_focus':
                if self.current_focus_filter:
                    self._add_to_journey_log("FILTER_CLEARED", "Focus filter cleared.", original_user_query_for_context, details=f"Previous filter: {self.current_focus_filter}")
                    self.current_focus_filter = None
                    responses.append({"type": "info", "message": "Focus cleared."})
                else:
                    responses.append({"type": "info", "message": "No focus active to clear."})

            elif command == '/whatif':
                if not args:
                    responses.append({"type": "info", "message": "Usage: /whatif [your 'what if' scenario question]"})
                else:
                    what_if_user_query = args.strip()
                    logging.info(f"Processing /whatif query: {what_if_user_query}")
                    # AI generates code for "what if"
                    what_if_code_gen_prompt = f"""The user's "what if" scenario query is:
"{what_if_user_query}"

The current DataFrame df (which might be filtered, current shape {current_df.shape}) is the one to be copied and modified.
The original unfiltered DataFrame has columns described by:
{self.data_context_string}

Your task is to generate Python code that:

Creates a copy of the current DataFrame df: df_copy = df.copy(). This is critical.

Applies the hypothetical scenario modifications described in the user's query to df_copy.

Calculates and print()s the specific impact or result the user is asking about (e.g., new total, change in a metric, count of rows meeting a new condition).
Ensure the print output is clear and directly answers the user's implied question about the impact.

Provide ONLY the Python code block. No explanations before or after.

IMPORTANT: Ensure the entire output is a single HTML block.
"""
                    ai_response_whatif_code = self._send_to_ai(what_if_code_gen_prompt, f"/whatif code gen for {what_if_user_query}")
                    what_if_code = self._extract_python_code(ai_response_whatif_code or "")

                    if not what_if_code:
                        responses.append({"type": "error", "message": "AI failed to generate Python code for the 'What If' scenario."})
                        self._add_to_journey_log("WHAT_IF_NO_CODE", f"AI failed to generate code for: {what_if_user_query}", what_if_user_query)
                    else:
                        self.pending_whatif_code_to_execute = {"code": what_if_code, "original_query": what_if_user_query}
                        responses.append({
                            "type": "whatif_code_ready",
                            "code": what_if_code,
                            "original_query": what_if_user_query,
                            "next_actions": ["/execute_pending_whatif_code"],
                            "message": "AI generated code for 'What If' scenario. Use /execute_pending_whatif_code to run it."
                        })
            
            elif command == '/execute_pending_whatif_code':
                if not self.pending_whatif_code_to_execute:
                    responses.append({"type": "error", "message": "No 'What If' code is pending execution. Use /whatif first."})
                else:
                    code_to_run = self.pending_whatif_code_to_execute["code"]
                    original_query = self.pending_whatif_code_to_execute["original_query"]
                    logging.info(f"Executing pending 'What If' code for query: {original_query}")
                    
                    # Execute on current_df context (AI was told to copy it)
                    success, scenario_output, error_msg = self._execute_generated_code(code_to_run, current_df, f"/whatif exec for {original_query}")

                    if not success:
                        responses.append({"type": "execution_error", "message": f"Error executing 'What If' code: {error_msg}", "code_attempted": code_to_run})
                        self._add_to_journey_log("WHAT_IF_EXEC_ERROR", f"Execution error for: {original_query}", original_query, details=f"Code:\n{code_to_run}\nError:\n{error_msg}")
                    else:
                        responses.append({"type": "whatif_execution_output", "output": scenario_output or "(No direct output from scenario code)", "original_query": original_query})
                        # Stage 3: AI explains the result
                        explanation_prompt = f"""The user asked the "what if" scenario: 
"{original_query}"

The Python code generated and executed for this scenario produced the following output:
--- BEGIN SCENARIO OUTPUT ---
{scenario_output}
--- END SCENARIO OUTPUT ---

Please explain this result to the user in a clear and concise way, relating it back to their original "what if" question.
Do not generate any Python code in this response. Focus only on the textual explanation.

IMPORTANT: Ensure the entire output is a single HTML block.
"""
                        ai_explanation = self._send_to_ai(explanation_prompt, f"/whatif explanation for {original_query}") or "AI explanation not available."
                        responses.append({"type": "whatif_ai_explanation", "explanation": ai_explanation, "original_query": original_query})
                        self._add_to_journey_log("WHAT_IF_EXPLORED", f"Scenario: {original_query[:80]}... Output: {str(scenario_output)[:50]}...", original_query, details=f"Code:\n{code_to_run}\nOutput:\n{scenario_output}\nExplanation:\n{ai_explanation}")

                        self.pending_whatif_code_to_execute = None # Clear after attempt

            elif command == '/history_summary':
                responses.append({"type": "info", "message": f"Current AI chat history length (user/model turns): {len(self.chat_session.history) // 2}. Max turns kept: {self.max_history_turns}."})
            
            elif command == '/clear_history':
                if len(self.chat_session.history) > 2: # System prompt and its ack
                    self.chat_session.history = self.chat_session.history[:2]
                    responses.append({"type": "info", "message": "Chat history cleared (system prompt retained)."})
                else:
                    responses.append({"type": "info", "message": "Chat history is already minimal."})
            
            elif command == '/journey':
                if self.analysis_journey_log:
                    responses.append({"type": "journey_log", "log": self.analysis_journey_log})
                else:
                    responses.append({"type": "info", "message": "No analysis journey recorded yet."})
            
            elif command == '/execute_pending_code':
                if not self.pending_code_to_execute:
                    responses.append({"type": "error", "message": "No general code is pending execution."})
                else:
                    code_info = self.pending_code_to_execute
                    self.pending_code_to_execute = None # Clear before execution
                    
                    logging.info(f"Executing pending code for query: {code_info['original_query']}")
                    execution_result = self._handle_execute_code_interaction(
                        code_info['code'], 
                        code_info['original_query'], 
                        code_info['is_plot'],
                        current_df
                    )
                    responses.extend(execution_result.get("responses", []))
                    if execution_result.get("status") == "pending_fix_confirmation":
                        # The _handle_execute_code_interaction already set self.pending_code_to_execute for the fix
                        pass # message already added by _handle_execute_code_interaction

            elif command == '/analyze_last_plot':
                if not self.last_executed_plot_path or not os.path.exists(self.last_executed_plot_path):
                    responses.append({"type": "error", "message": "No plot has been generated recently, or path is invalid."})
                else:
                    try:
                        img = Image.open(self.last_executed_plot_path)
                        plot_insight_prompt_parts: List[Union[str, Image.Image]] = [
                            f"""Please analyze the provided plot image.

The plot was generated in response to a user query (or a step in analyzing) roughly related to: '{original_user_query_for_context}' (this might be the command '/analyze_last_plot' itself, or the query that led to the plot).
The current DataFrame focus is: '{self.current_focus_filter if self.current_focus_filter else "None"}'.
Filename: {os.path.basename(self.last_executed_plot_path)}.

Provide detailed insights based only on what you can see in the image:

    What type of plot is it?

    What variables are shown?

    What are the key trends, patterns, or relationships visible?

    Are there any notable outliers or anomalies?

    What conclusions or further questions might arise from this visualization?
    Keep your analysis concise and focused on the visual information.
    
IMPORTANT: Ensure the entire output is a single HTML block.
""",
                            img
                        ]
                        ai_plot_insight_text = self._send_to_ai(plot_insight_prompt_parts, f"Plot analysis for {os.path.basename(self.last_executed_plot_path)}")

                        if ai_plot_insight_text:
                            responses.append({"type": "plot_analysis_result", "insights": ai_plot_insight_text, "plot_path": self.last_executed_plot_path})
                            summary = self._get_ai_summary_of_finding(ai_plot_insight_text, f"Plot analysis: {os.path.basename(self.last_executed_plot_path)}")
                            self._add_to_journey_log("PLOT_INSIGHT", summary or "Analyzed plot.", original_user_query_for_context, details=ai_plot_insight_text)
                        else:
                            responses.append({"type": "info", "message": "AI did not provide insights for the plot."})
                    except Exception as img_err:
                        logging.error(f"Error during plot analysis for {self.last_executed_plot_path}: {img_err}")
                        responses.append({"type": "error", "message": f"Could not analyze plot: {img_err}"})
        
            elif command == '/suggest_next_steps':
                if not self.analysis_journey_log:
                    responses.append({"type": "info", "message": "No analysis journey yet to base suggestions on. Try asking a question first."})
                else:
                    journey_summary = self._get_journey_summary_for_ai()
                    latest_finding_or_action = self.analysis_journey_log[-1]['description'] if self.analysis_journey_log else "the initial dataset exploration"
                    next_step_prompt = f"""Based on the following summary of the analysis journey:
{journey_summary}

And considering the latest finding/action was related to: "{latest_finding_or_action}"
(The user's last explicit query was: '{original_user_query_for_context}')

Suggest 2-3 logical next analytical questions or explorations that can be performed on the current DataFrame df.
Frame these as questions the user might ask, or direct suggestions for what to investigate next.
Be specific and actionable.

IMPORTANT: Ensure the entire output is a single HTML block.
"""
                    suggestions = self._send_to_ai(next_step_prompt, "Next step suggestions")
                    if suggestions:
                        responses.append({"type": "next_step_suggestions", "suggestions": suggestions})
                    else:
                        responses.append({"type": "info", "message": "AI could not provide next step suggestions at this time."})
            else:
                responses.append({"type": "error", "message": f"Unknown command: {command}. Try /help."})

    # --- NATURAL LANGUAGE QUERY (Not a command) ---
        else:
            prompt_for_ai = original_user_query_for_context
            if self.current_focus_filter:
                prompt_for_ai = f"Current Focus (already applied to `df`): {self.current_focus_filter}\nUser Query: {original_user_query_for_context}"
            
            # If image is attached, prepare multipart prompt
            ai_prompt_content: Union[str, List[Union[str, Image.Image]]]
            if image_attachment:
                # Make sure prompt_for_ai comes first for context
                ai_prompt_content = [prompt_for_ai, image_attachment]
                self._add_to_journey_log("IMAGE_QUERY", "User asked a question with an image.", original_user_query_for_context, details="Image attached.")
            else:
                ai_prompt_content = prompt_for_ai

            ai_response_text = self._send_to_ai(ai_prompt_content, original_user_query_for_context)

            if not ai_response_text:
                responses.append({"type": "error", "message": "AI did not provide a response."})
            else:
                initial_generated_code = self._extract_python_code(ai_response_text)
                text_part = ai_response_text
                if initial_generated_code: # Remove code from text_part if present
                    text_part = re.sub(r"```python\s*(.*?)\s*```", "", ai_response_text, flags=re.DOTALL | re.IGNORECASE).strip()

                if text_part:
                    responses.append({"type": "text_response", "content": text_part})
                
                if initial_generated_code:
                    is_plot_code = "plt." in initial_generated_code or "sns." in initial_generated_code
                    code_type_msg = "plotting" if is_plot_code else "calculation"
                    
                    if self.auto_execute_enabled:
                        responses.append({"type": "info", "message": f"AI generated {code_type_msg} code. Auto-executing..."})
                        execution_result = self._handle_execute_code_interaction(
                            initial_generated_code, 
                            original_user_query_for_context, 
                            is_plot_code,
                            current_df # Pass the current_df for execution
                        )
                        responses.extend(execution_result.get("responses", []))
                        if execution_result.get("status") == "pending_fix_confirmation":
                            pass # self.pending_code_to_execute already set by _handle_execute_code_interaction
                    else: # Auto-execute is OFF
                        self.pending_code_to_execute = {
                            "code": initial_generated_code, 
                            "original_query": original_user_query_for_context, 
                            "is_plot": is_plot_code
                        }
                        responses.append({
                            "type": "code_ready_for_execution",
                            "code": initial_generated_code,
                            "is_plot_code": is_plot_code,
                            "message": f"AI generated {code_type_msg} code. To execute, use /execute_pending_code.",
                            "next_actions": ["/execute_pending_code"]
                        })
                        self._add_to_journey_log("CODE_GENERATED_PENDING_EXEC", f"Generated {code_type_msg} code for: {original_user_query_for_context}", original_user_query_for_context, details=initial_generated_code)
                elif not text_part and not initial_generated_code: # AI response was empty after stripping potential code block markers
                    responses.append({"type": "info", "message": "AI provided an empty response."})


        # After processing, if no specific responses were added but it wasn't an error, add a generic ack.
        if not responses:
            responses.append({"type": "info", "message": "Query processed. No specific output to display."})
            
        return responses

if __name__ == '__main__':
    # This is a simple example of how to use the ChatbotService.
    # In a real application, you'd instantiate and use it within your app's structure (e.g., a web server).
    print("Initializing ChatbotService (this might take a moment for data loading and AI setup)...")

    # --- Basic Logging for the example ---
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_num_example = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level_num_example, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
    logger_example = logging.getLogger("ChatbotServiceExample")


    try:
        # You can override defaults here, e.g., pass your GOOGLE_API_KEY if not in .env
        # Ensure 'supermarket_sales.csv' is in the same directory or provide the correct path.
        csv_path = 'supermarket_sales.csv'
        if not os.path.exists(csv_path):
            logger_example.error(f"CRITICAL: CSV file '{csv_path}' not found. Please download or place it in the correct location.")
            print(f"\nError: '{csv_path}' not found. Please ensure the CSV file is present.")
            print("You can download it from sources like Kaggle (e.g., 'AUNG PAING PHYO/Supermarket sales').")
            exit(1)
            
        chatbot = ChatbotService(csv_file_path=csv_path)
        print("\n--- Advanced CSV Chatbot Service Ready ---")
        print(f"Data: '{os.path.basename(chatbot.csv_file_path)}' | Model: {chatbot.model_name}")
        print(f"Auto-execute is initially OFF. Use '/config auto_execute on' to enable.")
        print("Type /help for commands, or ask a question about the data. Type /quit to exit.\n")

        starter_qs = chatbot.get_starter_questions()
        if starter_qs:
            print("Here are some starter questions you could ask:")
            for q in starter_qs:
                print(f"  - {q}")
            print("-" * 30)

        while True:
            try:
                user_input = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chatbot example...")
                break
            
            if not user_input.strip():
                continue
            if user_input.lower() == '/quit':
                print("Exiting chatbot example...")
                break

            service_responses = chatbot.process_user_query(user_input)
            
            print("\n[Service Response(s)]")
            for res_item in service_responses:
                print(f"  Type: {res_item.get('type')}")
                if 'message' in res_item:
                    print(f"  Message: {res_item['message']}")
                if 'content' in res_item:
                    print(f"  Content: {res_item['content']}")
                if 'code' in res_item:
                    print(f"  Code:\n```python\n{res_item['code']}\n```")
                if 'path' in res_item: # For plot paths
                    print(f"  Plot Path: {res_item['path']}")
                if 'output' in res_item: # For calculation results or whatif output
                    print(f"  Output: {res_item['output']}")
                if 'explanation' in res_item: # For whatif explanation
                    print(f"  Explanation: {res_item['explanation']}")
                if 'insights' in res_item: # For plot analysis
                    print(f"  Insights: {res_item['insights']}")
                if 'questions' in res_item: # For starter questions
                    print("  Suggested Questions:")
                    for q_s in res_item['questions']: print(f"    - {q_s}")
                if 'log' in res_item: # For journey log
                    print("  Journey Log:")
                    for i, entry in enumerate(res_item['log']):
                        print(f"    {i+1}. [{entry['timestamp']}] ({entry['type']}) {entry['description']}")
                        if entry['details'] and len(entry['details']) < 150: print(f"       Details: {entry['details']}")
                        elif entry['details']: print(f"       Details: {entry['details'][:150]}...")
                if 'next_actions' in res_item and res_item['next_actions']:
                    print(f"  Suggested Next Commands: {', '.join(res_item['next_actions'])}")
                if 'impact_assessment' in res_item: # For focus proposal
                    print(f"  Impact Assessment:\n{res_item['impact_assessment']}")
                if 'ai_review' in res_item: # For focus proposal
                    print(f"  AI Review:\n{res_item['ai_review']}")
                if res_item.get("type") not in ["help_text", "journey_log", "data_context"] : # Avoid too much spacing for these
                    print("-" * 20) # Separator for individual response items
            print("-" * 30 + "[End Service Response(s)]\n")


    except ValueError as ve:
        logger_example.critical(f"Initialization failed: {ve}")
        print(f"Critical Error: {ve}")
    except Exception as e:
        logger_example.exception("An unexpected error occurred in the example runner:")
        print(f"An unexpected error occurred: {e}")