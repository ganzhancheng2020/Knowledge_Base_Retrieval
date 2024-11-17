from typing import Any
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from zhipuai import ZhipuAI
from dotenv import dotenv_values
from typing import ClassVar
import time
import asyncio

# from llama_index.legacy.llms.types import (
#     CompletionResponseAsyncGen
# )

from loguru import logger
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="INFO", rotation="00:00", retention="10 days", compression="zip")

class GLM(CustomLLM):
    config: dict = dotenv_values(".env")
    glm_api_key: ClassVar[str] = config["glm_api_key"]
    temperature: float = 0.7
    context_window: int = 8192
    num_output: int = 4096
    model: str = "glm-4-flash"
    
    def __init__(self, model="glm-4-flash"):
        """初始化 GLM 类，允许传入自定义 model_name。"""
        super().__init__(model=model)
        self.model = model

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:

        try:
            client = ZhipuAI(api_key=self.glm_api_key)
            response = client.chat.completions.create(
                model = self.model,
                temperature = self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            content = response.choices[0].message.content
            #logger.info(f"respond:{content}")
            return CompletionResponse(text=content, raw=response)
        except Exception as e:
            logger.error(f"Processing: {prompt} An error occurred during completion: {e}")
            return CompletionResponse(text="")
    
#     async def astream_complete(
#         self, prompt: str, formatted: bool = False, **kwargs: Any
#     ) -> CompletionResponseAsyncGen:
#         """Stream complete the prompt."""
#         if not formatted:
#             prompt = self.completion_to_prompt(prompt)

#         return await super().astream_complete(prompt, **kwargs)
#     @llm_completion_callback()
#     async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        
#         try:
#             client = ZhipuAI(api_key=self.glm_api_key)
#             response = await client.chat.asyncCompletions.create(
#                 model = self.model_name,
#                 temperature = self.temperature,
#                 messages=[
#                     {"role": "user", "content": prompt}
#                 ],
#             )
#             task_id = response.id
#             task_status = ''
#             get_cnt = 0
#             max_retries = 5
#             retry_interval = 2
#             while task_status != 'SUCCESS' and task_status != 'FAILED' and get_cnt < max_retries:
#                 result_response = client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
#                 task_status = result_response.task_status
#                 if task_status == 'SUCCESS':
#                     text = result_response.choices[0].message.content
#                     return CompletionResponse(text=text)
 
#                 if task_status == 'FAILED':
#                    logger.error(f"Task failed: {result_response.error_message}")
#                    return CompletionResponse(text="")
            
#                 await asyncio.sleep(retry_interval)
#                 get_cnt += 1
            
#             logger.error("Task timed out.")
#             return CompletionResponse(text="")
    
#         except Exception as e:
#             logger.error(f"Processing: {prompt} An error occurred during completion: {e}")
#             return CompletionResponse(text="")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        client = ZhipuAI(api_key=self.glm_api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        # 处理流式响应
        for chunk in response:
            delta = chunk.choices[0].delta  # 获取增量文本
            yield delta  # 使用生成器返回增量文本