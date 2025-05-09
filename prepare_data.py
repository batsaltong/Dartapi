import os
import requests
import zipfile

DART_API_KEY = os.getenv("DART_API_KEY")  # .env에 저장된 키를 쓰거나 직접 문자열로 입력

url = "https://opendart.fss.or.kr/api/corpCode.xml"
params = {'crtfc_key': DART_API_KEY}
response = requests.get(url, params=params)

# zip파일 저장
with open("CORPCODE.zip", "wb") as f:
    f.write(response.content)

# 압축 해제
with zipfile.ZipFile("CORPCODE.zip", "r") as zip_ref:
    zip_ref.extractall("./")  # 현재 폴더에 CORPCODE.xml 생성

os.remove("CORPCODE.zip")  # zip파일 삭제(선택)
print("CORPCODE.xml 파일이 생성되었습니다.")
