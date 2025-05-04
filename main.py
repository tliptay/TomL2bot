import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)


class TemplateForecaster(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = 2  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research_perplexity = ""
            research_metaculus = ""
            research_resolution_criteria = ""
            research_asknews = ""
            
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research_asknews = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            #elif os.getenv("PERPLEXITY_API_KEY"):
            #    research = await self._call_perplexity(question.question_text)
            
            if os.getenv("OPENROUTER_API_KEY"):
                research_perplexity = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
                research_metaculus = await self._call_perplexity_metaculus(
                    question.question_text, use_open_router=True
                )
                research_resolution_criteria = await self._call_perplexity_resolution_criteria(
                    question.resolution_criteria, question.question_text, use_open_router=True
                )
            else:
                logger.warning(
                    f"No research provider found when processing question URL {question.page_url}. Will pass back empty string."
                )
                research = ""

            research = research_asknews + "\n\n" + research_perplexity + "\n\n" + research_metaculus + "\n\n" + research_resolution_criteria
            
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )

            print(f'<RESEARCH>\n {research}\n </RESEARCH>\n')      
            
            return research

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster. 
            The superforecaster will give you a question they intend to forecast on. 
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news and information sources for helping them research the question. 
            When possible, try to get a diverse range of perspectives if the question is controversial. 
            Use your judgment in deciding the most relevant information. 
            You do not produce forecasts yourself - you are responsible for retrieving relevant information for the superforecaster.

            The question is:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _call_perplexity_resolution_criteria(
        self, resolution_criteria: str, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are a research assistant.
            
            Provide any information found from URLs listed in the text below that might be relevant for answering this question:
            {question}

            Only provide information directly from any URLs from the text:
            
            <text>
            {resolution_criteria}
            </text>
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        
        
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response
        
    async def _call_perplexity_metaculus(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are a research assistant.
            Find any open Metaculus questions that are similar to the question below. Do not provide any resolved or closed questions.
            Provide the forecasts on those questions and a brief summary.
            
            The question is: {question}
            """
        ) 
        
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        
        return response

    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            <background>
            {question.background_info}
            </background> 

            Please use the information and research gathered by your trusted assistant below:
            <research>
            {research}
            </research>

            This question's outcome will be determined by the specific resolution criteria below. Assume this criteria is not yet satisfied:
            <resolution criteria>
            {question.resolution_criteria}

            {question.fine_print}
            </resolution criteria>

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Tips from good forecasters:
            - If an event has been anticipated to occur in 3 months, but 2.5 months have gone by and it still hasn't happened yet and there is no recent news within the last week, then you should be skeptical that it will happen on the stated timeframe. It probably means that it will be delayed, or plans have changed.
            - Think about the base rates for similar events in the past.
            - Put extra weight on the status quo outcome since the world changes slowly most of the time.
            - Think about if there are seasonal effects.
            - Think about what the current trend is and if it makes sense to extrapolate, or not. Some things like stock prices are effectively random walks, so recent trends likely don't matter.
            - Even if something seems impossible, never forecast less than 5%. (It is possible that you don't have all of the information, or have misunderstood something.)
            - Even if something seems certain, never forecast more than 95%. (It is possible that you don't have all of the information, or have misunderstood something.)
            - Pay close attention to the exact wording and resolution source in the resolution criteria. Sometimes newspaper articles will cite a number that is significantly different from the number in the resolution criteria. Make sure to pay attention to the resolution criteria.
            - Like a good forecaster, you should use your own judgment to come to the most accurate forecast!

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.
            
            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=1, min_prediction=0
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            <background>
            {question.background_info}
            </background> 

            Please use the information and research provided by your trusted assistant below:
            <research>
            {research}
            </research>

            This question's outcome will be determined by the specific resolution criteria below. Assume this criteria is not yet satisfied:
            <resolution criteria>
            {question.resolution_criteria}

            {question.fine_print}
            </resolution criteria>

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            Tips from good forecasters:
            - Think about the base rates for similar events in the past.
            - Put extra weight on the status quo outcome since the world changes slowly most of the time. 
            - Think about if there are seasonal effects.
            - Think about what the current trend is and if it makes sense to extrapolate, or not. Some things like stock prices are effectively random walks, so recent trends likely don't matter.
            - Even if an option seems impossible, never put less than 2% on an option. (It is possible that you don't have all of the information, or have misunderstood something.)
            - Even if an option seems certain, never put more than 95% on an option. (It is possible that you don't have all of the information, or have misunderstood something.)            
            - Pay close attention to the exact wording and resolution source in the resolution criteria. Sometimes newspaper articles will cite a number that is significantly different from the number in the resolution criteria. Make sure to pay attention to the resolution criteria.
            - Like a good forecaster, you should use your own judgment to come to the most accurate forecast.

            The last thing you write is your final probabilities for the options in this order {question.options} as:
            Option A: Probability A %
            Option B: Probability B %
            ...
            Option N: Probability N %

            For each "Option X" in the list above, replace it with the actual option text from this list: {question.options}
            Do not write any text after the percent sign for your probability of an option.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        print(f'<REASONING> {reasoning} </REASONING>')
        
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )

        print(f'<PREDICTION>{prediction}</PREDICTION>/n')
        
        for p in prediction:
            print(f'<PREDICT>{p}</PREDICT>/n')
        
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            <background>
            {question.background_info}
            </background> 

            Please use the information and research provided by your trusted assistant below:
            <research>
            {research}
            </research>

            This question's outcome will be determined by the specific resolution criteria below. Assume this criteria is not yet satisfied:
            <resolution criteria>
            {question.resolution_criteria}

            {question.fine_print}
            </resolution criteria>

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) A brief description of an unexpected scenario that results in a low outcome.
            (e) A brief description of an unexpected scenario that results in a high outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters err on the side of having wide confidence intervals to account for unexpected outcomes.

            Important tips: 
            - Your 20th percentile forecast should be greater than the lower bound of {question.lower_bound}.
            - Your 80th percentile forecast should be less than the upper bound of {question.upper_bound}.
            - Have a wide range for your tails since you might not have all the information, or might be misunderstanding something.
            - Historically your 80/20 confidence interval has been much too narrow. To be more calibrated make your P10 to P90 4x wider than your P20 to P80 interval. These wide tails will help your calibration.
            - Think about if there are seasonal effects.
            - Think about what the current trend is and if it makes sense to extrapolate, or not. Some things like stock prices are effectively random walks, so recent trends likely don't matter.
            - Pay close attention to the exact wording and resolution source in the resolution criteria. Sometimes newspaper articles will cite a number that is significantly different from the number in the resolution criteria. Make sure to pay attention to the resolution criteria.
            - Like a good forecaster, use your own judgment!

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there.
            - For parsing purposes, don't refer to "percentile" in your reasoning. Instead, use P10, P90...
            - Only use the word "Percentile" in your final answer.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )

        print(f'<CONTINUOUS>\n {prediction.declared_percentiles}\n </CONTINUOUS>\n')
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        # llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
        #     "default": GeneralLlm(
        #         model="metaculus/anthropic/claude-3-5-sonnet-20241022",
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        # },
        llms={ # LLM models to use for different tasks. Will use default llms if not specified. Requires the relevant provider environment variables to be set.
            "default": GeneralLlm(
                model="openrouter/google/gemini-2.5-pro-preview-03-25",
                temperature=0.2,
            ),
            "summarizer": "openrouter/openai/gpt-4o-mini",
        }
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
               MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            # "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            # "https://www.metaculus.com/questions/21017/next-pope/",
            "https://www.metaculus.com/questions/37151/george-santos-sentence-length/",
            "https://www.metaculus.com/questions/34751/home-battery-annual-frequency-imbalance-return/",
            "https://www.metaculus.com/questions/36881/medicaid-cut-over-10-years-in-2025-reconciliation-bill/",
            "https://www.metaculus.com/questions/26327/us-measles-outbreak-2025/",
            "https://www.metaculus.com/questions/605/global-warming-in-2100-over-1880-baseline/",
            "https://www.metaculus.com/questions/9062/time-from-weak-agi-to-superintelligence/",
            "https://www.metaculus.com/questions/31817/h5-case-fatality-rate-in-us/",
            
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
