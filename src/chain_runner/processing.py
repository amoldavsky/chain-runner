import ast
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Callable, TypeVar, Iterable, Any, Optional

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai
from langchain.schema import (
    BaseMessage,
    SystemMessage
)
from langchain.chat_models.base import BaseChatModel
from langchain_openai.chat_models import (
    ChatOpenAI
)
import asyncio
import json
import dotenv


dotenv.load_dotenv()


# Custom callback function to be called on retry
def on_retry(retry_state):
    print(f"Error: {retry_state.outcome.exception()}. Retrying... Attempt: {retry_state.attempt_number}")

# Retry configuration: Exponential backoff with max 5 attempts
@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(openai.RateLimitError),
       after=on_retry)  # Call on_retry before each retry
def process_row_with_backoff(row, process_row):
    return process_row(row)



# -------------------------
# Generic async batch processing
# -------------------------

# Define generic type variables:
T = TypeVar('T')  # The entire data structure (e.g. a DataFrame, list, etc.)
U = TypeVar('U')  # A single item from the data structure

async def process_batch(
    data: T,
    iter_fn: Optional[Callable[[T], Iterable[U]]],  # A function that yields iterable items from data
    process_item: Callable[[U], Any],       # A function that processes a single item
    max_workers: int = 10
):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, process_row_with_backoff, item, process_item)
            for item in (iter_fn(data) if callable(iter_fn) else data)
        ]
        responses = await asyncio.gather(*tasks)
    return responses


def parse_json_obj(response: str) -> (dict, str):
    si = response.find("{")
    si = si if si > -1 else 0
    ei = response.rfind("}", si)
    data_str = response[si:ei + 1]
    data_json = json.loads(data_str)
    data_json_str = json.dumps(data_json)
    return data_json, data_json_str

# -------------------------
# JSON parsing helper functions
# -------------------------

def parse_json_arr(response: str) -> (dict, str):
    si = response.find("[")
    si = si if si > -1 else 0
    ei = response.rfind("]", si)
    data_str = response[si:ei + 1]
    data_json = json.loads(data_str)
    data_json_str = json.dumps(data_json)
    return data_json, data_json_str


def parse_json(response: str) -> (dict, str):
    si1 = response.find("[")
    if si1 < 0:
        return parse_json_obj(response)

    si2 = response.find("{")
    if si2 < 0:
        return parse_json_arr(response)

    if si1 < si2:
        return parse_json_arr(response)
    else:
        return parse_json_obj(response)


def process_row(
    messages: Union[
        Iterable[BaseMessage],
        Callable[[U], Iterable[BaseMessage]] # generator function
    ],
    on_invoke: Optional[Callable[[Any], Any]] = None,
    response_fn: Optional[Callable[[str, U, Any], Any]] = None,
    chat: BaseChatModel = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=1.0,
        top_p=1.0
    )
) -> Callable[[U], Any]:
    """
    Helper function to "replay" conversations through a copy of the Shopper
    agent, given a map of shopper_id to prompt text.

    Args:
        messages (Union[str, Callable]): if a generator function will be called otherwise treated as str
        response_fn: response transforming function
        on_invoke: callback function that will be invoked before Langchain invoke.
        chat: instance of BaseChatModel
    Returns:

    """

    _messages = messages
    def is_iterable_of_basemessage(x: Any) -> bool:
        if not isinstance(x, Iterable):
            return False
        # Exclude strings and bytes
        if isinstance(x, (str, bytes)):
            return False
        return all(isinstance(item, BaseMessage) for item in x)
    if not (is_iterable_of_basemessage(messages) or callable(messages)):
        raise TypeError("messages must be a generator function, or list of BaseMessage.")

    def _process_row(row: U) -> Any:
        # generate chat context
        messages = _messages
        if callable(_messages):
            messages = _messages(row)
            if not is_iterable_of_basemessage(messages):
                raise TypeError("messages must be a generator function, or list of BaseMessage.")
        messages = messages or []

        try:
            if callable(on_invoke):
                on_invoke({row, messages, chat})

            # TODO: add progress bar
            response_raw = chat.invoke(
                messages
            )
            response = response_raw.content.strip()
            if response_fn is not None:
                return response_fn(response, row, {
                    "messages": messages,
                    "response_raw": response_raw,
                })

            return response
        except Exception as e:
            # an error, we really probably don't care
            print("error: ", e)
            return None
    return _process_row
