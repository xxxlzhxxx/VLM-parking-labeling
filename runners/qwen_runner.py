import concurrent.futures
import os
import base64
import requests
from openai import OpenAI


class QwenRunner:
    def __init__(self, api_key, model_id, base_url=None):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self._image_cache = {}  # cache base64 by URL

    def _get_image_base64(self, image_url):
        """Download image and convert to base64 data URL for Qwen API."""
        if image_url in self._image_cache:
            return self._image_cache[image_url]

        # Check local cache first
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        image_name = os.path.basename(image_url)
        local_path = os.path.join(script_dir, 'data', 'images', image_name)

        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                image_data = f.read()
        else:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            image_data = resp.content

        b64 = base64.b64encode(image_data).decode('utf-8')
        # Detect mime type from extension
        ext = os.path.splitext(image_name)[1].lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        mime = mime_map.get(ext, 'image/jpeg')
        data_url = f"data:{mime};base64,{b64}"

        self._image_cache[image_url] = data_url
        return data_url

    def _invoke(self, prompt_text, image_url):
        # Convert image to base64 data URL since Qwen may not access TOS URLs
        data_url = self._get_image_base64(image_url)

        return self.client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            extra_body={
                "enable_thinking": True,
                "vl_high_resolution_images": True
            },
            temperature=0.1,
            top_p=0.7
        )

    def run(self, prompt_text, image_url, timeout_seconds):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._invoke, prompt_text, image_url)
            return future.result(timeout=timeout_seconds)
