# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.

from typing import List, Optional

import torch
from transformers import AutoTokenizer, TextIteratorStreamer


class BatchTextIteratorStreamer(TextIteratorStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from accessing the generated text in a non-blocking way (e.g., in an interactive
    Gradio demo).

    This iterator extends TextIteratorStreamer to support batching of text generation. Each put operation appends
    generated text to a batch, and the end operation finalizes the batch by processing and storing the generated
    sequences.

    Parameters:
        batch_size (int):
            The size of each text generation batch.
        tokenizer (AutoTokenizer):
            The tokenizer used to decode the tokens.
        skip_prompt (bool, optional, default=False):
            Whether to skip the prompt to `.generate()` or not. Useful, for example, for chatbots.
        timeout (float, optional):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()` when it is called in a separate thread.
        **decode_kwargs (dict, optional):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    """

    def __init__(
        self,
        batch_size: int,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.batch_size: int = batch_size
        self.token_cache: List[List[int]] = [[] for _ in range(batch_size)]
        self.print_len = [0] * batch_size
        self.blocked = False

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to buffer as soon as they form entire words.
        """

        if len(value.shape) < 2:
            value = torch.reshape(value, (self.batch_size, value.shape[0] // self.batch_size))

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        batch_printable_text = []
        for i in range(self.batch_size):
            # Add the new token to the cache and decodes the entire thing
            self.token_cache[i].extend(value[i].tolist())
            text = self.tokenizer.decode(self.token_cache[i], **self.decode_kwargs)

            # After the symbol for a new line, we flush the cache.
            if text.endswith("\n"):
                printable_text = text[self.print_len[i] :]
                self.token_cache[i] = []
                self.print_len[i] = 0
            # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len[i] :]
                self.print_len[i] += len(printable_text)
            # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
            # which may change with the subsequent token -- there are probably smarter ways to do this!)
            else:
                printable_text = text[self.print_len[i] : text.rfind(" ") + 1]
                self.print_len[i] += len(printable_text)
            batch_printable_text.append(printable_text)

        self.on_finalized_text(batch_printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        batch_printable_text = []
        for idx in range(self.batch_size):
            if len(self.token_cache[idx]) > 0:
                text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
            else:
                printable_text = ""
            batch_printable_text.append(printable_text)

        self.next_tokens_are_prompt = True
        self.on_finalized_text(batch_printable_text, stream_end=True)
        self.blocked = False

    def on_finalized_text(self, texts: List[str], stream_end: bool = False):
        self.text_queue.put(texts, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    # thkim change for demo
    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value

    def block(self):
        self.blocked = True

    def is_blocked(self):
        return self.blocked
