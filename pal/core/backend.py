# Copyright 2022 PAL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import pdb

from dotenv import load_dotenv
load_dotenv()

class LanguageModelBackend:
    """
    General backend for language models, with batching and retry logic.
    Child classes must implement _completions_api and _chat_api.
    """
    def call_gpt(self, prompt, model:str='code-davinci-002', stop=None, temperature=0., top_p=1.0,
                 max_tokens=128, majority_at=None):
        num_completions = majority_at if majority_at is not None else 1
        num_completions_batch_size = 5
        completions = []
        for i in range(20 * (num_completions // num_completions_batch_size + 1)):
            try:
                requested_completions = min(num_completions_batch_size, num_completions - len(completions))
                if self._is_chat_model(model):
                    ans = self._chat_api(
                        model=model,
                        max_tokens=max_tokens,
                        stop=stop,
                        prompt=prompt,
                        temperature=temperature,
                        top_p=top_p,
                        n=requested_completions,
                        best_of=requested_completions)
                else:
                    ans = self._completions_api(
                        model=model,
                        max_tokens=max_tokens,
                        stop=stop,
                        prompt=prompt,
                        temperature=temperature,
                        top_p=top_p,
                        n=requested_completions,
                        best_of=requested_completions)
                completions.extend(ans)
                if len(completions) >= num_completions:
                    return completions[:num_completions]
            except Exception as e:
                self._handle_rate_limit(e, i)
        raise RuntimeError('Failed to call language model API')

    def call_chat_gpt(self, messages, model='gpt-3.5-turbo', stop=None, temperature=0., top_p=1.0, max_tokens=128):
        wait = 1
        while True:
            try:
                return self._call_chat_gpt(messages, model, stop, temperature, top_p, max_tokens)
            except Exception as e:
                self._handle_rate_limit(e, wait)
                wait *= 2
        raise RuntimeError('Failed to call chat gpt')

    def _call_chat_gpt(self, messages, model, stop, temperature, top_p, max_tokens):
        # Child class should override this if needed
        raise NotImplementedError

    def _completions_api(self, model, max_tokens, stop, prompt, temperature,
                        top_p, n, best_of):
        raise NotImplementedError

    def _chat_api(self, model, max_tokens, stop, prompt, temperature,
                top_p, n, best_of):
        raise NotImplementedError

    def _is_chat_model(self, model:str):
        raise NotImplementedError
        

    def _handle_rate_limit(self, error, wait):
        # Default: sleep on rate limit errors, otherwise re-raise
        if hasattr(error, 'status_code') and error.status_code == 429:
            time.sleep(min(wait, 60))
        elif hasattr(error, 'error') and getattr(error.error, 'type', None) == 'rate_limit_error':
            time.sleep(min(wait, 60))
        elif 'Exceeded monthly credit' in str(error):
            time.sleep(min(wait, 180))
        else:
            raise error

class OpenAIBackend(LanguageModelBackend):
    def __init__(self):
        import openai
        self.openai = openai
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.openai.api_key = self.api_key
        self.tokens_used = 0

    def _completions_api(self, model, max_tokens, stop, prompt, temperature,
                        top_p, n, best_of):
        ans = self.openai.Completion.create(
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            n=n,
            best_of=best_of)
        self.tokens_used += ans['usage']['total_tokens']
        return [choice['text'] for choice in ans['choices']]

    def _chat_api(self, model, max_tokens, stop, prompt, temperature,
                top_p, n, best_of):
        ans = self.openai.ChatCompletion.create(
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant that can write Python code that solves mathematical reasoning questions similarly to the examples that you will be provided.'},
                {'role': 'user', 'content': prompt}],
            temperature=temperature,
            top_p=top_p,
            n=n)
        return [choice['message']['content'] for choice in ans['choices']]

    def _call_chat_gpt(self, messages, model, stop, temperature, top_p, max_tokens):
        ans = self.openai.ChatCompletion.create(
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            n=1
        )
        return ans.choices[0]['message']['content']

    def _is_chat_model(self, model):
        return model.startswith('gpt-4') or model.startswith('gpt-3.5-turbo')

class HuggingFaceBackend(LanguageModelBackend):
    def __init__(self):
        import requests
        self.requests = requests
        self.API_URL = "https://router.huggingface.co/v1/chat/completions"
        self.API_TOKEN = os.environ.get('HUGGINGFACE_API_KEY')

        self.headers = {
            "Authorization": f"Bearer {self.API_TOKEN}",
        }
        self.tokens_used = 0

    def _completions_api(self, model, max_tokens, stop, prompt, temperature, top_p, n, best_of):
        prompt += " /nothink"
        payload = {
            "model": model,
            "messages": [
                {'role': 'system', 'content': 'You are a helpful assistant that can write Python code that solves mathematical reasoning questions similarly to the examples that you will be provided.'},
                {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop_sequences": stop,
            "best_of": best_of
        }
        response = self.requests.post(self.API_URL, headers=self.headers,
                                       json=payload)
        if response.status_code == 200:
            data = response.json()
            self.tokens_used += data['usage']['total_tokens']
            return [choice['message']['content'] for choice in data['choices']]
        elif response.status_code == 402:
            error = Exception(f"Exceeded monthly credit, end of inference loop")
            self._handle_rate_limit(error, 120)
        else:
            error = Exception(f"Request failed with status code {response.status_code}: {response.text}")
            self._handle_rate_limit(error, 1)
    
    def _chat_api(self, model, max_tokens, stop, prompt, temperature, top_p, n, best_of):
        payload = {
            "model": model,
            "messages": [
                {'role': 'system', 'content': 'You are a helpful assistant that can write Python code that solves mathematical reasoning questions similarly to the examples that you will be provided.'},
                {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop_sequences": stop,
            "best_of": best_of
        }
        response = self.requests.post(self.API_URL, headers=self.headers,
                                       json=payload)
        if response.status_code == 200:
            data = response.json()
            return [choice['message']['content'] for choice in data['choices']]
        else:
            error = Exception(f"Request failed with status code {response.status_code}: {response.text}")
            self._handle_rate_limit(error, 1)
    
    def _call_chat_gpt(self, messages, model, stop, temperature, top_p, max_tokens):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stop_sequences": stop,
            "temperature": temperature,
            "top_p": top_p,
            "n": 1
        }
        response = self.requests.post(self.API_URL, headers=self.headers,
                                       json=payload)
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            error = Exception(f"Request failed with status code {response.status_code}: {response.text}")
            self._handle_rate_limit(error, 1)

    def _is_chat_model(self, model):
        return model.find("Instruct") != -1

# Example usage:
# backend = OpenAIBackend()
# result = backend.call_gpt("your prompt here")