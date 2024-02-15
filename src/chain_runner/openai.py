from dotenv import load_dotenv
import openai
import os
import backoff
import json
import tiktoken

# take environment variables from .env.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def on_backoff(e):
    print('target: {}, tries: {}, exception: {}'.format(e['target'], e['tries'], e['exception']))


@backoff.on_exception(backoff.expo, Exception, max_tries=10, on_backoff=on_backoff, factor=2)
def ask_gpt(messages, *args, **kwargs):
    # defaults + overrides
    kwargs = {**{
        "max_tokens": 2048,
        "temperature": 0.2
    }, **kwargs}

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        *args, **kwargs
    )
    answer: str = response['choices'][0]['message']['content']

    input_tokens, input_cost = estimate_price_gpt4_input(" ".join([m["role"] + " " + m["content"] for m in messages]))
    output_tokens, output_cost = estimate_price_gpt4_output(answer)
    stats = {
        "input_tokens": input_tokens,
        "input_cost": input_cost,
        "output_tokens": output_tokens,
        "output_cost": output_cost,
        "tokens": input_tokens + output_tokens,
        "cost": input_cost + output_cost
    }

    # # Extract tokens used from the API response
    # tokens_count = response['usage']['total_tokens']
    # print("GPT tokens: ", tokens_count)

    return answer, stats, response


# async def ask_gpt_async(messages, *args, **kwargs):
#     # Retry with exponential backoff
#     for attempt in backoff.on_exception(backoff.expo, Exception, max_tries=5):
#         with attempt:
#             return ask_gpt(messages, *args, **kwargs)

def parse_json_obj(response: str) -> (dict, str):
    si = response.find("{")
    si = si if si > -1 else 0
    ei = response.rfind("}", si)
    data_str = response[si:ei + 1]
    data_json = json.loads(data_str)
    data_json_str = json.dumps(data_json)
    return data_json, data_json_str

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


def estimate_price_gpt4_input(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(enc.encode(text))
    price_per_token = 0.03 / 1000
    return num_tokens, num_tokens * price_per_token


def estimate_price_gpt4_output(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(enc.encode(text))
    price_per_token = 0.06 / 1000
    return num_tokens, num_tokens * price_per_token
