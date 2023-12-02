from typing import List, Union, Tuple, Dict, Generator

import nltk
from fastapi import UploadFile, File, Form, Body
from langchain_core.documents import Document
from pydantic import Json
from config.config import *

from langchain_file_spliter.utils import (
    get_file_path,
    BaseResponse,
    KnowledgeFile,
)

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


def _save_files_in_thread(
    files: List[UploadFile], knowledge_base_name: str, override: bool
):
    """
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        """
        保存单个文件。
        """
        try:
            filename = file.filename
            file_path = get_file_path(
                knowledge_base_name=knowledge_base_name, doc_name=filename
            )
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()  # 读取上传文件的内容
            if (
                os.path.isfile(file_path)
                and not override
                and os.path.getsize(file_path) == len(file_content)
            ):
                # TODO: filesize 不同后的处理
                file_status = f"文件 {filename} 已存在。"
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(
                f"{e.__class__.__name__}: {msg}", exc_info=e if log_verbose else None
            )
            return dict(code=500, msg=msg, data=data)

    params = [
        {"file": file, "knowledge_base_name": knowledge_base_name, "override": override}
        for file in files
    ]
    for file in files:
        result = save_file(
            file=file, knowledge_base_name=knowledge_base_name, override=override
        )
        yield result


def files2docs_in_thread(
    files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = OVERLAP_SIZE,
    zh_title_enhance: bool = ZH_TITLE_ENHANCE,
) -> Generator:
    """
    利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    """

    def file2docs(
        file: KnowledgeFile, **kwargs
    ) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
            logger.error(
                f"{e.__class__.__name__}: {msg}", exc_info=e if log_verbose else None
            )
            return False, (file.kb_name, file.filename, msg)

    # kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            # kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            result = file2docs(file, **kwargs)
            yield result
        except Exception as e:
            yield False, (kb_name, filename, str(e))


def update_docs(
    knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
    file_names: List[str] = Body(
        ..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]
    ),
    chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
    chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
    zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
    override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
    docs: Json = Body(
        {},
        description="自定义的docs，需要转为json字符串",
        examples=[{"test.txt": [Document(page_content="custom doc")]}],
    ),
    not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    更新知识库文档
    """

    failed_files = {}
    kb_files = []

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        # file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        # if file_detail.get("custom_docs") and not override_custom_docs:
        #     continue
        if file_name not in docs:
            try:
                kb_files.append(
                    KnowledgeFile(
                        filename=file_name, knowledge_base_name=knowledge_base_name
                    )
                )
            except Exception as e:
                msg = f"加载文档 {file_name} 时出错：{e}"
                logger.error(
                    f"{e.__class__.__name__}: {msg}",
                    exc_info=e if log_verbose else None,
                )
                failed_files[file_name] = msg

    # 从文件生成docs，并进行向量化。
    # 这里利用了KnowledgeFile的缓存功能，在多线程中加载Document，然后传给KnowledgeFile
    for status, result in files2docs_in_thread(
        kb_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        zh_title_enhance=zh_title_enhance,
    ):
        if status:
            kb_name, file_name, new_docs = result
            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            kb_file.splited_docs = new_docs
            # kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # 将自定义的docs进行向量化
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            # kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(
                f"{e.__class__.__name__}: {msg}", exc_info=e if log_verbose else None
            )
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        # kb.save_vector_store()
        pass

    return BaseResponse(code=200, msg=f"更新文档完成", data={"failed_files": failed_files})


def upload_docs(
    files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
    knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
    override: bool = Form(False, description="覆盖已有文件"),
    to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
    chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
    chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
    zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
    docs: Json = Form(
        {},
        description="自定义的docs，需要转为json字符串",
        examples=[{"test.txt": [Document(page_content="custom doc")]}],
    ),
    not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
):
    """
    API接口：上传文件，并/或向量化
    """

    failed_files = {}
    file_names = list(docs.keys())

    # 先将上传的文件保存到磁盘
    for result in _save_files_in_thread(
        files, knowledge_base_name=knowledge_base_name, override=override
    ):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # 对保存的文件进行向量化
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            # 保存到向量库
            pass

    return BaseResponse(code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files})
