import os

import json
from langchain.tools.base import BaseTool
from langchain.schema.output_parser import BaseOutputParser
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import load_prompt

from langchain_core.prompts import PipelinePromptTemplate, BasePromptTemplate
import tempfile

import sys
sys.path.append("..")
from ..agent.Action import *





def chinese_friendly(string) -> str:
    lines = string.split("\n")
    for i ,line in enumerate(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii = False)
            except:
                pass
        return "\n".join(lines)
    

def load_file(filename: str, encoding = "utf-8") -> str:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"file {filename} not found.")
    f = open(filename, "r", encoding=encoding)
    filestr = f.read()
    f.close()
    return filestr


class PromptTemplateBuilder:
    def __init__(
            self,
            prompt_path:str,
            prompt_file:str,
    ):
        self.prompt_file = prompt_file
        self.prompt_path = prompt_path

    def _get_tools_prompt(self, tools:List[BaseTool]) -> str:
        tools_prompt = ""
        for i, tool in enumerate(tools):
            prompt = f"{i + 1}.{tool.name}:{tool.description},\
                    args json schema: {json.dumps(tools.args, ensure_ascii= False)}\n"
            tools_prompt += prompt
        return tools_prompt
    
    """相对路径转换为绝对路径"""
    def _check_or_redirect(self, prompt_file: str) -> str:
        with open(prompt_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if "template_path" in config:
            # 如果是相对路径，则转换为绝对路径
            if not os.path.isabs(config["template_path"]):
                config["template_path"] = os.path.join(self.prompt_path, config["template_path"])
                # 生成临时文件
                tmp_file = tempfile.NamedTemporaryFile(
                    suffix='.json',
                    mode="w",
                    encoding="utf-8",
                    delete=False
                )
                tmp_file.write(json.dumps(config, ensure_ascii=False))
                tmp_file.close()
                return tmp_file.name
        return prompt_file
    
    def build(
            self,
            tools:Optional[List[BaseTool]] = None,
            output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:

        main_file = os.path.join(self.prompt_path, self.prompt_file)
        main_prompt_template = load_prompt(
            self._check_or_redirect(main_file)
        )
        variables = main_prompt_template.input_variables
        partial_variables = {}
        recursive_templates = []

        # 遍历所有变量，检查是否存在对应的模板文件
        for var in variables:
            # 是否存在嵌套模板
            if os.path.exists(os.path.join(self.prompt_path, f"{var}.json")):
                sub_template = PromptTemplateBuilder(
                    self.prompt_path, f"{var}.json"
                ).build(tools=tools, output_parser=output_parser)
                recursive_templates.append((var, sub_template))
            # 是否存在文本文件
            elif os.path.exists(os.path.join(self.prompt_path, f"{var}.txt")):
                var_str = load_file(os.path.join(self.prompt_path, f"{var}.txt"))
                partial_variables[var] = var_str

        if tools is not None and "tools" in variables:
            tools_prompt = self._get_tools_prompt(tools)
            partial_variables["tools"] = tools_prompt

        if output_parser is not None and "format_instructions" in variables:
            partial_variables["format_instructions"] = chinese_friendly(
                output_parser.get_format_instructions()
            )

        if recursive_templates:
            # 将有值嵌套的模板填充到主模板中
            main_prompt_template = PipelinePromptTemplate(
                final_prompt=main_prompt_template,
                pipeline_prompts=recursive_templates
            )

        # 将有值的变量填充到模板中
        main_prompt_template = main_prompt_template.partial(**partial_variables)

        return main_prompt_template


if __name__ == "__main__":
    # 创建一个PromptTemplateBuilder实例，指定文件路径和json文件路径
    builder = PromptTemplateBuilder("../prompts/main", "main.json")
    # 创建一个PydanticOutputParser实例，指定pydantic对象
    output_parser = PydanticOutputParser(pydantic_object=Action)
    # 使用builder实例和output_parser实例构建一个PromptTemplate实例
    prompt_template = builder.build(tools=[], output_parser=output_parser)
    # 打印出构建的PromptTemplate实例，并传入参数
    print(prompt_template.format(
        ai_name="瓜瓜",
        ai_role="智能助手机器人",
        task_description="解决问题",
        work_dir=".",
        short_term_memory="",
        long_term_memory="",
        tools="",
    ))