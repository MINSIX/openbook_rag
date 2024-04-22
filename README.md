# RAG를 이용해 오픈북 시험치기
pdf파일이나 excel 파일을 읽어와서 오픈북 시험치기

시험 있는줄 모르고 있다가 오픈북이라서 급하게 만든거


가지고 있는 데이터 형식에 따라 PDFLOADER부분을 EXCEL을 읽어오는 코드나, 데이터베이스 읽어오는 코드 등으로 수정하면됩니다.

엑셀의 경우 context부분이 형식이 달라서 수정좀해줘야합니다.

max_new_token 크기를 줄이면 생성 시간이 줄어듭니다.

Causal모델은 자기가 쓰고 싶은걸로 쓰시면 됩니다.

파인튜닝한 데이터가 있으면 
```
# model = PeftModel.from_pretrained(
#         model,
#         "./LDCC",
#         offload_dir=".",
#         torch_dtype=torch.float16,
#     )
```
이거 추가해서 쓰시면 됩니다. 성능이 유의미한 차이가 있을거라 하는데 저는 안해봤습니다.

