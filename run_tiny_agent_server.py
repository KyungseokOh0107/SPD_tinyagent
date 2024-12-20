import asyncio
import os
import signal
from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.tiny_agent.config import get_tiny_agent_config
from src.tiny_agent.models import (
    LLM_ERROR_TOKEN,
    TINY_AGENT_DIR,
    ModelType,
    streaming_queue,
)
from src.tiny_agent.tiny_agent import TinyAgent
from src.tiny_agent.transcription import (
    TranscriptionService,
    WhisperCppClient,
    WhisperOpenAIClient,
)
from src.utils.logger_utils import enable_logging, enable_logging_to_file, log
from src.utils.custom_logger_utils import CustomLogger
import time
enable_logging(False)
enable_logging_to_file(True)

CONFIG_PATH = os.path.join(TINY_AGENT_DIR, "Configuration.json")

app = FastAPI()


def empty_queue(q: asyncio.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except asyncio.QueueEmpty:
            # Handle the case where the queue is already empty
            break


class TinyAgentRequest(BaseModel):
    query: str

class TinyAgentRequest_EXPERIMENT(BaseModel):
    query: str
    path: str
    experiment_id: int = 0
    planner_answer: str = None
    generation_token_list: list[int] = [None, None]

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    """
    Custom error handling for logging the errors to the TinyAgent log file.
    """
    log(f"HTTPException {exc.status_code}: {exc.detail}")
    return PlainTextResponse(exc.detail, status_code=exc.status_code)


@app.post("/generate")
async def execute_command(request: TinyAgentRequest) -> StreamingResponse:
    """
    This is the main endpoint that calls the TinyAgent to generate a response to the given query.
    """
    custom_logger = CustomLogger(PATH = '/home/munyeolpark/spd/TinyAgent/')

    e2e_time_start = time.time()
    print(f"[SYSTEM] E2E_START_TIME: {0:.4f}")
    log(f"\n\n====\nReceived request: {request.query}")
    # First, ensure the queue is empty
    empty_queue(streaming_queue)

    query = request.query

    if not query or len(query) <= 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="No query provided"
        )
    try:
        tiny_agent_config = get_tiny_agent_config(config_path=CONFIG_PATH)
        tiny_agent = TinyAgent(tiny_agent_config, e2e_time_start, custom_logger)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error: {e}",
        )

    async def generate():
        try:
            response_task = asyncio.create_task(tiny_agent.arun(query))

            while True:
                # Await a small timeout to periodically check if the task is done
                try:
                    token = await asyncio.wait_for(streaming_queue.get(), timeout=1.0)
                    if token is None:
                        break
                    if token.startswith(LLM_ERROR_TOKEN):
                        raise Exception(token[len(LLM_ERROR_TOKEN) :])
                    yield token
                except asyncio.TimeoutError:
                    pass  # No new token, check task status

                # Check if the task is done to handle any potential exception
                if response_task.done():
                    break

            # Task created with asyncio.create_task() do not propagate exceptions
            # to the calling context. Instead, the exception remains encapsulated within
            # the task object itself until the task is awaited or its result is explicitly retrieved.
            # Hence, we check here if the task has an exception set by awaiting it, which will
            # raise the exception if it exists. If it doesn't, we just yield the result.
            await response_task
            response = response_task.result()
            yield f"\n\n{response}"
        except Exception as e:
            # You cannot raise HTTPExceptions in an async generator, it doesn't
            # get caught by the FastAPI exception handling middleware. Hence,
            # we are manually catching the exceptions and yielding/logging them.
            yield f"Error: {e}"
            log(f"Error: {e}")

        e2e_time_end = time.time() - e2e_time_start
        print(f"[SYSTEM] E2E_END_TIME: {e2e_time_end:.4f}")
        print(f'END')
        custom_logger.update_e2e_time(0, e2e_time_end)
        custom_logger.save_logging_result()
        custom_logger.save_tool_time()

    return StreamingResponse(generate(), media_type="text/event-stream")
# async def execute_command(request: TinyAgentRequest) -> StreamingResponse:
    """
    This is the main endpoint that calls the TinyAgent to generate a response to the given query.
    """
    log(f"\n\n====\nReceived request: {request.query}")

    # First, ensure the queue is empty
    empty_queue(streaming_queue)

    query = request.query

    if not query or len(query) <= 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="No query provided"
        )

    try:
        tiny_agent_config = get_tiny_agent_config(config_path=CONFIG_PATH)
        tiny_agent = TinyAgent(tiny_agent_config)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error: {e}",
        )

    async def generate():
        try:
            response_task = asyncio.create_task(tiny_agent.arun(query))

            while True:
                # Await a small timeout to periodically check if the task is done
                try:
                    token = await asyncio.wait_for(streaming_queue.get(), timeout=1.0)
                    if token is None:
                        break
                    if token.startswith(LLM_ERROR_TOKEN):
                        raise Exception(token[len(LLM_ERROR_TOKEN) :])
                    yield token
                except asyncio.TimeoutError:
                    pass  # No new token, check task status

                # Check if the task is done to handle any potential exception
                if response_task.done():
                    break

            # Task created with asyncio.create_task() do not propagate exceptions
            # to the calling context. Instead, the exception remains encapsulated within
            # the task object itself until the task is awaited or its result is explicitly retrieved.
            # Hence, we check here if the task has an exception set by awaiting it, which will
            # raise the exception if it exists. If it doesn't, we just yield the result.
            await response_task
            response = response_task.result()
            yield f"\n\n{response}"
        except Exception as e:
            # You cannot raise HTTPExceptions in an async generator, it doesn't
            # get caught by the FastAPI exception handling middleware. Hence,
            # we are manually catching the exceptions and yielding/logging them.
            yield f"Error: {e}"
            log(f"Error: {e}")

    return StreamingResponse(generate(), media_type="text/event-stream")



@app.post("/generate_experiment")
async def execute_command(request: TinyAgentRequest_EXPERIMENT) -> StreamingResponse:
    """
    This is the main endpoint that calls the TinyAgent to generate a response to the given query.
    """
    if request.planner_answer == 'None':
        request.planner_answer = None
    
    if request.generation_token_list == [0, 0]:
        request.generation_token_list = [None, None]

    custom_logger = CustomLogger(file_dir=request.path,
                                 experiment_id=request.experiment_id,
                                 planner_answer=request.planner_answer,
                                 generation_token_list=request.generation_token_list)

    e2e_time_start = time.time()
    print(f"[SYSTEM] E2E_START_TIME: {0:.4f}")
    log(f"\n\n====\nReceived request: {request.query}")
    # First, ensure the queue is empty
    empty_queue(streaming_queue)

    query = request.query

    if not query or len(query) <= 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="No query provided"
        )
    
    init_time_start = time.time() - e2e_time_start
    print(f"[SYSTEM] INIT_START_TIME: {init_time_start:.4f}")
    try:
        tiny_agent_config = get_tiny_agent_config(config_path=CONFIG_PATH)
        tiny_agent = TinyAgent(tiny_agent_config, e2e_time_start, custom_logger)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error: {e}",
        )
    init_time_end = time.time() - e2e_time_start
    print(f"[SYSTEM] INIT_END_TIME: {init_time_end:.4f}")
    custom_logger.update_init_time(init_time_start, init_time_end)

    async def generate():
        try:
            response_task = asyncio.create_task(tiny_agent.arun(query))

            while True:
                # Await a small timeout to periodically check if the task is done
                try:
                    token = await asyncio.wait_for(streaming_queue.get(), timeout=1.0)
                    if token is None:
                        break
                    if token.startswith(LLM_ERROR_TOKEN):
                        raise Exception(token[len(LLM_ERROR_TOKEN) :])
                    yield token
                except asyncio.TimeoutError:
                    pass  # No new token, check task status

                # Check if the task is done to handle any potential exception
                if response_task.done():
                    break

            # Task created with asyncio.create_task() do not propagate exceptions
            # to the calling context. Instead, the exception remains encapsulated within
            # the task object itself until the task is awaited or its result is explicitly retrieved.
            # Hence, we check here if the task has an exception set by awaiting it, which will
            # raise the exception if it exists. If it doesn't, we just yield the result.
            await response_task
            response = response_task.result()
            yield f"\n\n{response}"
        except Exception as e:
            # You cannot raise HTTPExceptions in an async generator, it doesn't
            # get caught by the FastAPI exception handling middleware. Hence,
            # we are manually catching the exceptions and yielding/logging them.
            yield f"Error: {e}"
            log(f"Error: {e}")

        e2e_time_end = time.time() - e2e_time_start
        print(f"[SYSTEM] E2E_END_TIME: {e2e_time_end:.4f}")
        print(f'END')
        custom_logger.update_e2e_time(0, e2e_time_end)
        custom_logger.save_logging_result()
        custom_logger.save_tool_time()

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/voice")
async def get_voice_transcription(request: Request) -> Response:
    """
    This endpoint call whisper to get voice transcription. It takes in bytes of audio
    returns the transcription in plain text.
    """
    log("\n\n====\nReceived request to get voice transcription")

    body = await request.form()
    audio_file = cast(UploadFile, body["audio_pcm"])
    sample_rate = int(cast(str, body["sample_rate"]))
    raw_bytes = await audio_file.read()

    if not raw_bytes or len(raw_bytes) <= 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="No audio provided"
        )
    if not sample_rate:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="No sampling rate provided"
        )

    try:
        tiny_agent_config = get_tiny_agent_config(config_path=CONFIG_PATH)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error: {e}",
        )

    whisper_client = (
        WhisperOpenAIClient(tiny_agent_config)
        if tiny_agent_config.whisper_config.provider == ModelType.OPENAI
        else WhisperCppClient(tiny_agent_config)
    )

    transcription_service = TranscriptionService(whisper_client)

    try:
        transcription = await transcription_service.transcribe(raw_bytes, sample_rate)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error: {e}",
        )

    return Response(transcription, status_code=HTTPStatus.OK)


@app.post("/quit")
async def shutdown_server() -> Response:
    """
    Shuts down the server by sending a SIGINT signal to the main process,
    which is a gentle way to terminate the server. This endpoint should be
    protected in real applications to prevent unauthorized shutdowns.
    """
    os.kill(os.getpid(), signal.SIGTERM)
    return Response("Server is shutting down...", status_code=HTTPStatus.OK)


@app.get("/ping")
async def ping() -> Response:
    """
    A simple endpoint to check if the server is running.
    """
    return Response("pong", status_code=HTTPStatus.OK)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=51000)