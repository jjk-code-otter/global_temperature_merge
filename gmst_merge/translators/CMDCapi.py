import os
import time
import requests
from urllib.parse import quote
import re


class CMDCClient:
    def __init__(
        self,
        user_id,
        output_dir="./",
        poll_interval=10,
        max_poll=200,
        timeout=30
    ):
        self.order_api = "https://ai.data.cma.cn/aiApi/order/addCraOrder"
        self.query_api = "https://ai.data.cma.cn/aiApi/order/getOrderById"

        self.user_id = user_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.poll_interval = poll_interval
        self.max_poll = max_poll
        self.timeout = timeout

    @staticmethod
    def encode_params(params):
        """对所有参数进行URL编码"""
        return {k: quote(str(v)) for k, v in params.items()}

    def create_order(self, params):
        """创建订单"""
        params["userId"] = self.user_id
        p = self.encode_params(params)

        url = self.order_api + "?" + "&".join([f"{k}={v}" for k, v in p.items()])

        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()

        if js.get("code") != "200":
            raise RuntimeError(f"创建订单失败: {js}")

        order_id = js["data"]
        print(f"订单创建成功")
        return order_id

    def get_order_info(self, order_id):
        """查询订单状态"""
        url = f"{self.query_api}?id={order_id}&userId={self.user_id}"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()

        if js.get("code") != "200":
            raise RuntimeError(f"查询订单失败: {js}")

        return js["data"]

    def wait_order(self, order_id):
        """轮询到订单完成"""
        print(f"等待订单完成...")

        for i in range(self.max_poll):
            data = self.get_order_info(order_id)
            status = data["status"]
            msg = data.get("message", "")

            print(f"轮询 {i+1}/{self.max_poll} | 状态={status} | 信息={msg}")

            if status == 4:
                print("订单已准备好下载")
                return data
            elif status == 3:
                print("订单处理失败")
                raise RuntimeError(f"订单处理失败: {msg}")

            time.sleep(self.poll_interval)

        raise TimeoutError("订单处理超时")

    def download_file(self, url, order_id):
        """下载文件"""
        print(f"开始下载")

        r = requests.get(url, stream=True, timeout=self.timeout)
        r.raise_for_status()

        # 从header获取文件名
        cd = r.headers.get("Content-Disposition", "")
        file_name = None

        if "filename=" in cd:
            # 兼容filename="xxx.zip"
            file_name = re.findall(r'filename="?([^";]+)"?', cd)
            if file_name:
                file_name = file_name[0]

        # fallback: 从URL推断文件名
        if not file_name:
            file_name = url.split("/")[-1]
            # 如果没有扩展名，添加.dat
            if "." not in file_name:
                file_name = f"{order_id}.dat"

        save_path = os.path.join(self.output_dir, file_name)

        # 写入文件 + 下载进度条
        total = int(r.headers.get('content-length', 0))
        downloaded = 0

        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = downloaded * 100 // total
                        print(f"下载进度: {percent}% ", end="")

        print(f"下载完成: {save_path}")
        return save_path

    def retrieve(self, params):
        """主流程: 创建订单 → 轮询订单状态 → 下载"""

        # 创建订单
        order_id = self.create_order(params)

        # 等待任务完成
        data = self.wait_order(order_id)

        # 下载文件
        dl_url = data["fileDownloadUri"]

        file_path = self.download_file(dl_url, order_id)

        print(f"文件已保存：{file_path}")
        return file_path
