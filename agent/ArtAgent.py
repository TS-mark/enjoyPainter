from typing import List, Optional, Tuple

from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import ValidationError

from AutoAgent.Action import Action
from utils.printColorful import *
# from Utils.PrintUtils import *


class ArtAgent:
    def __init__(
            self,
            llm:BaseLLM,
            prompts_path:str,
            tools: List[BaseTool],
            work_dir:str = "./data",
            main_prompt_file: str = "main.json",
            final_prompt_file: str = "final_step.json",
            ai_name: Optional[str] = "小涂",
            ai_role: Optional[str] = "手绘教师，可以根据学生的知识提供专业的知识以及教授学生如何解决绘画问题，并根据工具列表提供相关的图像帮助",
            max_thought_steps:Optional[int] = 10,
            memory_retriever: Optional[VectorStoreRetriever] = None,
    ):
        self.llm = llm
        self.prompts_path = prompts_path
        self.tools = tools
        self.work_dir = work_dir
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.max_thought_steps = max_thought_steps
        self.memory_retriever = memory_retriever
        
        # 如果输出格式不正确，则尝试修复
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)

        self.main_prompt_file = main_prompt_file
        self.final_prompt_file = final_prompt_file

        def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
            for tool in self.tools:
                if tool.name == tool_name:
                    return tool
            return None
        

        def _step(self,
                  reason_chain,
                  task_description,
                  short_term_memory,
                  long_term_memory,
                  verbose = False
                  )->Tuple[Action, str]:
            """执行进一步思考"""

            response = ""
            for s in reason_chain.stream({
                "short_term_memory": short_term_memory.load_memory_variables({})["history"],
                "long_term_memory": long_term_memory.load_memory_variables(
                    {"prompt":task_description}
                    )['history'] if long_term_memory is not None else "",
            }):
                if verbose:
                    color_print(s, THOUGHT_COLOR, end = "")
                response += s
            
            action = self.robust_parser.parse(response)
            return (action, response)
        
        def _final_step(self, short_term_memory, task_description) -> str:
            finish_prompt = PromptTemplateBuilder(
                self.prompts_path,
                self.final_prompt_file,
            )