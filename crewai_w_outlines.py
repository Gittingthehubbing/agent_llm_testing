import json
from crewai import Agent, Task
import os
from crewai import Crew, Process
import llama_cpp
from pydantic import BaseModel, Field, conlist
from langchain_openai import ChatOpenAI
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from outlines.models import LlamaCpp as LlamaCpp_o
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
import outlines

from rich.console import Console


class CustomLangchain(BaseChatModel):
    lcpp_model: LlamaCpp_o
    tokenizer: Any
    model_name: str
    json_schema: Optional[str] = None
    mode: str
    max_tokens: int
    seed: int

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        console = Console()
        console.print(f"Received prompt: {messages}")

        if isinstance(messages, list):
            prompt = [{"role": m.type, "content": m.content} for m in messages]
            prompt = self.tokenizer.hf_tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False
            ).replace(self.tokenizer.hf_tokenizer.bos_token, "")
        console.print(f"Changed prompt to: {prompt}")
        print_style = "bold white on blue"
        console.print(f"Number of tokens in prompt: {len(self.tokenizer.tokenize(prompt.encode()))}", style=print_style)
        if self.mode == "json":
            generator = outlines.generate.json(self.lcpp_model, self.json_schema)
        else:
            generator = outlines.generate.text(self.lcpp_model)
        outstr = ""
        tokens = generator.stream(
            prompt,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        console.print("Starting output \n\n")
        for token in tokens:
            console.print(token, end="")
            outstr += token
        message = AIMessage(content=outstr, additional_kwargs={}, response_metadata={"time_in_seconds": 3})
        console.print("\nOutput done \n\n", style=print_style)
        console.print(f"Number of tokens in result: {len(self.tokenizer.tokenize(outstr.encode()))}", style=print_style)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "custom_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}


from transformers import AutoTokenizer
from llama_cpp.llama import Llama


os.environ["OPENAI_API_KEY"] = "xxx"
default_llm = ChatOpenAI(model="not_used", base_url="http://localhost:8080/v1")

gguf_path = "../model_ckpts/Hermes-3-Llama-3.1-8B.Q5_K_M.gguf"
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B", trust_remote_code=True, fast=False)
tokenizer = llama_cpp.llama_tokenizer.LlamaHFTokenizer(tokenizer)
model = Llama(
    model_path=gguf_path,
    tokenizer=tokenizer,
    n_ctx=8000,
    verbose=False,
    n_gpu_layers=-1,
    n_threads=0,
    flash_attn=True,
)
llm = outlines.models.LlamaCpp(model)


class StepsOverview(BaseModel):
    list_of_steps: conlist(str, min_length=3, max_length=5) = Field(..., description="A list of all steps required.")


SEED = 42001
planning_llm = CustomLangchain(
    lcpp_model=llm,
    tokenizer=tokenizer,
    model_name="lcpp",
    json_schema=json.dumps(StepsOverview.model_json_schema()),
    mode="json",
    max_tokens=500,
    seed=SEED,
)


research_llm = CustomLangchain(
    lcpp_model=llm,
    tokenizer=tokenizer,
    model_name="lcpp",
    mode="text",
    max_tokens=500,
    seed=SEED,
)

planning_agent = Agent(
    role="Senior Planner",
    goal="Create a plan",
    backstory="You are responsible for planning. You always output valid JSON.",
    allow_delegation=False,
    verbose=True,
    llm=planning_llm,
    # max_iter=5,
    # max_retry_limit=5,
)

# Create a researcher agent
researcher = Agent(
    role="Senior Researcher",
    goal="Discover groundbreaking technologies",
    verbose=True,
    llm=research_llm,
    backstory="A curious mind fascinated by cutting-edge innovation and the potential to change the world, you know everything about tech.",
    allow_delegation=False
)

make_plan = Task(
    description="Make a list of steps",
    expected_output="A list of steps",
    agent=planning_agent,
    # pydantic=StepsOverview,
)

# Task for the researcher
research_task = Task(
    description="Identify the next big trend in AI",
    agent=researcher,  # Assigning the task to the researcher
    expected_output="Your Final answer must be the full python code, only the python code and nothing else.",
)


# Instantiate your crew
tech_crew = Crew(
    agents=[planning_agent, researcher],
    tasks=[make_plan, research_task],
    process=Process.sequential,  # Tasks will be executed one after the other
)

# Begin the task execution
tech_crew.kickoff()
