import os
import json
from openai import OpenAI
import base64
from typing import Any, Dict, List, Union, Optional

from app.core.config import settings

# --- Helper function to encode image (Identical to original) ---
def encode_image_to_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}. Cannot include image.")
        return None
    except Exception as e:
        print(f"Warning: Error encoding image {image_path}: {e}")
        return None

def get_model_performance_prompt(perf_string: str) -> str:
    return f"""
    Analyze the following machine learning model performance comparison table. The table structure is defined by 'columns' and model results are in 'data'. Metrics might be for regression (like MAE, MSE, RMSE, R2 - lower error is better, R2 closer to 1 is better) or classification (like Accuracy, Precision, Recall, F1, AUC - generally higher is better). 'TT (Sec)' is training time in seconds (lower is better).

    ```json
    {perf_string}
    ```

    Provide a **concise summary** for a non-technical or semi-technical audience, explaining the key findings and recommending next steps. Output in HTML format.

    **Instructions for Analysis:**
    1.  **Identify Top Performers:** Determine which 1-2 models perform best based on the primary performance metrics present (e.g., lowest RMSE/MAE, highest R2 for regression; highest Accuracy/F1/AUC for classification). Mention the key metric(s) used for comparison.
    2.  **Consider Performance vs. Speed Trade-off:**
        * Highlight any models that offer a good balance (e.g., slightly lower performance but significantly faster training time `TT (Sec)`).
        * Point out if the top-performing models are particularly slow to train.
    3.  **Note Performance Tiers/Similarities:** Are there groups of models with very similar performance? Are simpler models (like Linear/Ridge Regression) competitive with more complex ones (like Random Forest, Gradient Boosting)?
    4.  **Mention Metric Consistency (if applicable):** Do different key metrics generally agree on the best model(s)? If a model excels in one metric but performs poorly in another relevant one, briefly note it.
    5.  **Be Concise and Clear:** Avoid overly technical jargon. Explain results in terms of "better accuracy," "lower prediction error," "faster training," etc.

    **Instructions for Recommendations:**
    **Propose Actionable Next Steps:** use a recommend and friendly tone to suggest concrete actions such as:
        * Suggest Candidates for Deployment/Further Work: Based on performance and speed, recommend 1-2 models as strong candidates for potential deployment or more intensive tuning.
        * "Further tune the hyperparameters of [Top Model Name(s)] to potentially improve performance."
        * "Investigate why [Specific Model] performs surprisingly well/poorly compared to others."
        * "Consider [Model Name] if training speed is a critical factor, despite slightly lower performance."
        * "Evaluate models based on the specific business metric that matters most before making a final decision."
        * (If performance is poor overall): "Consider exploring different feature engineering techniques or model architectures."

    **Output Format:**
    * A brief introductory sentence about comparing model performance.
    * Key findings regarding top performers and trade-offs (bullet points preferred).
    * Actionable recommendations and next steps (bullet points preferred).
    Ensure the entire output is a single HTML block.
    """

def get_summary_stats_prompt(stats_string: str) -> str:
    return f"""
    Analyze the following statistical summary data derived from a dataset:

    ```json
    {stats_string}
    ```

    Provide a **concise summary** (around 2-3 key bullet points or a short paragraph) for a non-technical person. Output in HTML format.

    **Instructions:**
    1.  **Be Brief:** Do not simply list all statistics for every column.
    2.  **Highlight Key Insights:** Focus on what's most *interesting* or *significant*. Look for:
        * **Dominant Categories:** Is one category ('top') much more frequent ('freq') than others (like in 'Area')?
        * **Wide Variations:** Are the numbers spread out? (e.g., large difference between 'min'/'max', or a high 'std' relative to the 'mean' for things like Sales or Marketing Spend). Explain this simply (e.g., "Sales figures vary quite a lot").
        * **Typical Values:** Briefly mention averages ('mean') or middle values ('50%') for important numerical columns like Sales.
        * **Anything Unusual:** Point out anything that seems noteworthy (e.g., very low minimum Marketing Spend).
    3.  **Suggest Next Steps:** use a recommend and friendly tone and based *only* on this summary, propose 1-2 specific, actionable questions or areas for further investigation that arise from the highlighted insights. Frame these as logical next steps someone might take. Examples: "Investigate why South America is the dominant area," "Explore the reasons behind the wide range in Sales," "Analyze the relationship between Marketing Spend and Sales."

    **Output Format:**
    * A brief introductory sentence.
    * Key insights (bullet points preferred).
    * Suggested and recommendation for next steps/questions (bullet points preferred).
    Ensure the entire output is a single HTML block.
    """

def get_correlation_matrix_prompt(corr_string: str) -> str:
    return f"""
    Analyze the following correlation matrix. This matrix shows how different numerical variables tend to move together (correlation coefficient from -1 to +1).

    ```json
    {corr_string}
    ```

    Provide a **concise summary** for a non-technical person, focusing on **genuinely interesting or unexpected findings** and suggesting actionable next steps. Avoid stating the obvious. Output in HTML format.

    **Instructions for Identifying Insights:**
    1.  **Prioritize Non-Obvious Strong Correlations:** Focus on strong positive (> 0.6 or 0.7) or strong negative (< -0.6 or -0.7) correlations between variables that represent **distinct concepts**.
        * *Do not mention* highly expected strong correlations. These are less insightful.
        * *Highlight* strong correlations between variables that are *not* directly related by definition.
    2.  **Highlight Significant Negative Correlations:** These often reveal interesting trade-offs or inverse relationships. Explain the potential implication simply.
    3.  **Identify Surprisingly Weak Correlations:** Point out correlations that are close to zero where a moderate or strong relationship might have been expected based on the variable names (e.g., between a spending variable and an outcome variable, if applicable). This lack of relationship can be as insightful as a strong one.
    4.  **Look for Clusters (Optional):** If multiple distinct variables are all highly correlated with each other, mention this potential grouping or shared underlying factor.
    5.  **Be Selective:** Do not list every correlation. Focus on the 2-4 most insightful findings based on the criteria above. Ignore self-correlations (1.0).

    **Instructions for Explaining and Next Steps:**
    1.  **Explain Simply:** Describe the relationship (e.g., "Variable A and Variable B tend to strongly increase together," or "Variable C tends to decrease when Variable D increases").
    2.  **Suggest Actionable Next Steps:** use a recommend and friendly tone. For each key insight, propose 1-2 specific questions or investigation areas. These should aim to understand *why* the relationship (or lack thereof) exists or what the business implications are. Frame them generally. Examples:
        * "Investigate the underlying reason for the unexpected strong link between [Variable X] and [Variable Y]."
        * "Explore factors that might explain the surprisingly weak relationship between [Variable P] and [Variable Q]."
        * "Analyze the potential trade-off indicated by the negative correlation between [Variable M] and [Variable N]."
        * "Examine if the cluster of correlated variables ([V1], [V2], [V3]) represents a common driver or influence."

    **Output Format:**
    * The most insightful relationships found (bullet points preferred).
    * Suggested next steps/questions based on those insights (bullet points preferred).
    Ensure the entire output is a single HTML block.
    """

def get_tuned_model_with_image_prompt_text(tuning_results_string: str) -> str:
    return f"""
    You have two pieces of information about an optimized prediction model:
    1.  **Tuning Performance Data (text below):** Results from testing the model after finding its best settings. This includes average performance and consistency across several tests.
        ```json
        {tuning_results_string}
        ```
    2.  **Feature Importance Plot (image):** An image you will also analyze, showing which input data characteristics (features) the model uses most to make predictions.

    **Your Task for a Non-Technical User:**
    Skip basic definitions. Directly provide valuable and interesting insights from both the tuning data and the feature importance image. Focus on what these findings mean practically and suggest clear, actionable next steps. Output in HTML format.

    **1. Key Findings from Model's Performance Tests (from JSON data):**
    * **How well did it perform on average?** Look at the 'Mean' row in the `cv_metrics_table`. Highlight a key performance indicator (e.g., for regression, focus on 'RMSE' or 'MAE' as "average prediction error," or 'R2' as "accuracy in explaining outcomes." For classification, it would be 'Accuracy', 'F1', etc.). State the average value simply.
        * *Insight example:* "On average, the model's predictions had an error of about [Mean RMSE value]." or "The model was able to explain about [Mean R2 value * 100]% of the outcomes."
    * **How consistent was its performance?** Look at the 'Std' (Standard Deviation) row for that same key metric.
        * *Insight example:* "This performance was [very consistent / fairly consistent / showed some noticeable variation] across the different tests (consistency score: [Std value])." (A low Std relative to the Mean is more consistent).

    **2. Key Factors Driving the Model's Predictions (from the Feature Importance Image):**
    * **What are the top 2-3 game-changers?** Based on your analysis of the image, identify the most influential features.
        * *Insight example:* "The model relies most heavily on [Top Feature 1] and [Top Feature 2] to make its predictions. [Feature 3] also plays a significant role."
    * **Are there any surprises among these key factors?**
        * *Insight example:* "It's interesting that [Surprising Top Feature] is so important. Does this give you a new perspective?" or "As might be expected, [Obvious Top Feature] is a key driver."
    * **Which factors had little impact?** Briefly mention if any commonly considered factors are shown to be unimportant by the model.
        * *Insight example:* "Interestingly, factors like [Less Important Feature 1] and [Less Important Feature 2] didn't have much influence on the model's predictions in this setup."

    **3. Actionable Recommendations & Next Steps (Synthesize Everything):** use a recommend and friendly tone to:
    * **Based on Performance:**
        * If good average performance & good consistency: "This optimized model performs well and reliably. It appears ready for [e.g., pilot testing, deployment in a controlled environment, use for specific decisions]."
        * If good average performance & poor consistency: "While the model shows good average performance, its results varied a bit across tests. If high reliability is critical for every prediction, consider [e.g., further investigation into the variability, testing on more diverse data before full deployment]."
        * If performance is borderline or needs improvement: "The current performance is [describe simply]. To improve it, you might consider [e.g., collecting more data on key features, trying different model types, further refining features]."
    * **Leveraging Feature Insights:**
        * "Focus on ensuring the data for the key drivers ([Top Feature 1], [Top Feature 2]) is always high quality and accurate, as they heavily influence the results."
        * "Discuss with your team: How can you use the knowledge that [Top Feature 1] and [Top Feature 2] are so influential? Does it change any business strategies or data collection priorities?"
    * **Specific Next Steps to Consider:**
        * "Validate these key features: Confirm with experts if the importance of these features makes sense and if it uncovers anything new."
        * "Monitor performance: If you decide to use this model, keep an eye on how well it performs on new, real-world data."
        * "Iterate if needed: Based on its real-world performance and your business goals, you can always revisit and further refine the model or features."

    Keep your language direct, simple, and focused on practical value. Answer as briefly as possible. Ensure the entire output is a single HTML block.
    Return <body>Your Answer</body>
    """
    
def get_baseline_model_with_image_prompt_text(metrics_string: str) -> str:
    """
    Returns a shorter textual part of the prompt for summarizing a general baseline model's
    performance (task type to be identified from metrics) and a feature importance image.
    This text is intended for a multimodal model like GPT-4o that will also see the image.
    """
    return f"""
    You have two pieces of information about a **baseline predictive model**:
    1.  **Performance Metrics (text below):** Key performance scores for the model.
        ```json
        {metrics_string}
        ```
    2.  **Feature Importance Plot (image):** An image you will also analyze, showing which input data characteristics (features) the model uses most to make predictions.

    **Your Task for a Non-Technical or Semi-Technical User (Output in HTML format):**

    1.  **Identify Model Task & Summarize Performance (from JSON data):**
        *   First, examine the metric names in the JSON to determine if this is a **classification** model (e.g., metrics like 'Accuracy', 'AUC', 'F1-score') or a **regression** model (e.g., metrics like 'R2', 'MAE', 'RMSE'). Clearly state the identified task type.
        *   Based on the identified task type, highlight the **key performance indicators** from the JSON and explain what they mean in simple terms. For example, if it's classification, you might focus on accuracy and AUC. If it's regression, you might focus on R-squared and an error metric like MAE or RMSE.
        *   Provide a brief overall assessment of this baseline performance (e.g., "This baseline model provides an initial benchmark. Its performance for [identified task] is characterized by [mention a key finding].").

    2.  **Analyze Key Predictive Factors (from the Feature Importance Image):**
        *   Identify the top 2-3 most influential features for this baseline model.
        *   Comment briefly on whether these top features are expected or surprising for this initial model.
        *   Mention if any features seem to have less impact than perhaps anticipated.

    3.  **Provide Actionable Recommendations & Next Steps (Friendly tone, emphasizing it's a baseline):**
        *   **Evaluate Baseline Adequacy:** "Based on its performance for [identified task] (e.g., [mention a key identified metric and its value]), is this baseline model's performance acceptable as a starting point, or is significant improvement the primary goal?"
        *   **Leverage Feature Insights:**
            *   "The most influential features ([Top Feature 1], [Top Feature 2]) are critical. Focus on ensuring their data quality and explore them further in subsequent models."
            *   "If a highly influential feature is unexpected, investigate why. If an expected important feature is *not* influential in this baseline, this could point to data issues or an opportunity for feature engineering."
        *   **Next Steps for Model Improvement:**
            *   "To improve upon this baseline for its [identified task], common next steps include exploring different algorithms suitable for this task, refining features (feature engineering), or addressing potential data issues (like imbalances for classification or outliers for regression)."
            *   "It's crucial to compare any more complex models against this baseline to quantify the benefits of increased complexity and effort."

    Keep your language direct, simple, and focused on practical value. Answer as briefly as possible. Ensure the entire output is a single HTML block.
    Return <body>Your Answer</body>
    """
    
class AISummaryService:
    # This was the default model in the original get_ai_summary function signature
    ORIGINAL_DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key not provided to AISummaryService.")
        self.client = OpenAI(api_key=api_key)

    def get_ai_summary(
        self,
        data: Any,
        input_type: str
    ) -> str:
        text_prompt_content = None
        system_message = "" # Will be set based on input_type
        temperature = 0.4
        max_tokens_default = 500
        max_tokens_multimodal_or_perf = 700 # For more complex/longer outputs

        messages: List[Dict[str, Any]] = []

        # Determine the primary model candidate based on override or original script's default
        primary_model_candidate = self.ORIGINAL_DEFAULT_MODEL

        if input_type == 'summary_stats':
            serializable_data = [item.model_dump(by_alias=True) for item in data]
            data_string = json.dumps(serializable_data, indent=2)
            text_prompt_content = get_summary_stats_prompt(data_string)
            system_message = "You are a helpful and friendly AI assistant skilled at extracting key insights from statistical summaries and suggesting next steps in simple terms. Format the output in HTML format."
            final_model_name = primary_model_candidate
            current_max_tokens = max_tokens_default
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text_prompt_content}
            ]
        elif input_type == 'correlation_matrix':
            data_string = json.dumps(data, indent=2)
            text_prompt_content = get_correlation_matrix_prompt(data_string)
            system_message = "You are a helpful and friendly AI assistant skilled at explaining correlation matrices, focusing on non-obvious insights and their implications in simple terms. Format the output in HTML format."
            final_model_name = primary_model_candidate or "gpt-3.5-turbo"
            current_max_tokens = max_tokens_default
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text_prompt_content}
            ]
        elif input_type == 'model_performance':
            data_string = json.dumps(data, indent=2)
            text_prompt_content = get_model_performance_prompt(data_string)
            system_message = "You are a helpful and friendly AI assistant skilled at interpreting machine learning model performance results and providing actionable recommendations in simple terms. Format the output in HTML format."
            final_model_name = primary_model_candidate or "gpt-3.5-turbo"
            current_max_tokens = max_tokens_multimodal_or_perf
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text_prompt_content}
            ]
        elif input_type == 'tuned_model_with_image_eval':
            if not isinstance(data, dict) or "tuning_data" not in data:
                return "Error: For 'tuned_model_with_image_eval', data must be a dictionary with 'tuning_data'."

            tuning_results_string = json.dumps(data["tuning_data"], indent=2)
            text_prompt_content = get_tuned_model_with_image_prompt_text(tuning_results_string)
            system_message = "You are an AI assistant skilled at explaining complex model tuning results by synthesizing textual data and visual feature importance plots for a non-technical audience. Format the output in HTML format."
            final_model_name = primary_model_candidate or "gpt-4o" # GPT-4o is a sensible fallback for multimodal
            current_max_tokens = max_tokens_multimodal_or_perf

            user_content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt_content}]

            base64_image = data.get("image_base64")
            mime_type = data.get("image_mime_type")

            if base64_image and mime_type:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })
            elif "image_url" in data and data["image_url"]:
                 user_content.append({"type": "image_url", "image_url": {"url": data["image_url"]}})
            # else: No change to text_prompt_content if image is missing, mirroring original script's behavior
            # where it would just send the text part without explicit mention of missing image in the prompt.

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
        elif input_type == 'baseline_model_with_image_eval':
            if not isinstance(data, dict) or "metrics_data" not in data:
                return "Error: For 'baseline_model_with_image_eval', data must be a dictionary with 'metrics_data'."
            
            metrics_data_string = json.dumps(data["metrics_data"], indent=2) # metrics_data is a dict
            text_prompt_content = get_baseline_model_with_image_prompt_text(metrics_data_string)
            system_message = "You are an AI assistant skilled at interpreting baseline model metrics and feature importance plots for a non-technical audience, providing actionable insights. Format the output in HTML format."
            final_model_name = primary_model_candidate or "gpt-4o" # Multimodal, so GPT-4o or similar is good
            current_max_tokens = max_tokens_multimodal_or_perf

            user_content: List[Dict[str, Any]] = [{"type": "text", "text": text_prompt_content}]
            base64_image = data.get("image_base64")
            mime_type = data.get("image_mime_type")

            if base64_image and mime_type:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
        else:
            return f"Error: Invalid input_type '{input_type}'."

        if not messages: # Should not happen if input_type is valid
             return "Error: Could not construct messages for API call."
        if not final_model_name: # Should be set by logic above using ORIGINAL_DEFAULT_MODEL
            # This is an internal fallback, should ideally not be reached if ORIGINAL_DEFAULT_MODEL is set.
            return "Error: Model name could not be determined (internal error)."


        try:
            response = self.client.chat.completions.create(
                model=final_model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=current_max_tokens
            )
            content = response.choices[0].message.content
            return content.strip() if content else "Error: Empty response from AI."
        except Exception as e:
            # Log the actual error for debugging on the server side
            print(f"Error during OpenAI API call for input_type {input_type} with model {final_model_name}: {e}")
            return f"Error during OpenAI API call: {str(e)}"


ai_summary_service_instance = AISummaryService(api_key=settings.OPENAI_API_KEY)