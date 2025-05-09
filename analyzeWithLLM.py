import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import xml.etree.ElementTree as ET

# 환경 변수 로드 (.env 파일)
load_dotenv()
DART_API_KEY = os.getenv("DART_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # LangChain에 필요한 API 키

if not DART_API_KEY:
    raise Exception("DART_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")

# CORPCODE.xml 파싱 (프로젝트 폴더에 미리 다운로드해 둘 것)
@st.cache_data
def load_corp_code_dict(xml_path='CORPCODE.xml'):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    corp_dict = {}
    for company in root.findall('list'):
        corp_name = company.find('corp_name').text.strip()
        corp_code = company.find('corp_code').text.strip()
        corp_dict[corp_name] = corp_code
    return corp_dict

corp_dict = load_corp_code_dict()

def get_stock_code(user_input):
    # 숫자 입력(고유번호) 지원
    if user_input.strip().isdigit():
        return user_input.strip()
    # 정확히 일치하는 기업명
    if user_input in corp_dict:
        return corp_dict[user_input]
    # 일부만 입력한 경우(포함 검색, 복수 결과 시 선택)
    matches = [name for name in corp_dict if user_input in name]
    if len(matches) == 1:
        return corp_dict[matches[0]]
    elif len(matches) > 1:
        st.warning(f"여러 기업이 검색되었습니다: {matches}. 정확한 기업명을 입력하세요.")
        raise Exception("여러 기업명이 검색됨")
    else:
        raise Exception("회사 이름을 찾을 수 없습니다. DART에 등록된 기업명인지 확인하세요.")

def fetch_dart_financials_df(company_code, bsns_year="2022", reprt_code="11011"):
    base_url = "https://opendart.fss.or.kr/api"
    endpoint = "fnlttSinglAcnt.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": company_code,
        "bsns_year": bsns_year,
        "reprt_code": reprt_code,
    }
    response = requests.get(f"{base_url}/{endpoint}", params=params)
    if response.status_code != 200:
        raise Exception("DART API 호출 실패: " + str(response.status_code))

    data = response.json()
    st.write("전체 API 응답:", data)

    financial_list = data.get("list", [])
    if not financial_list:
        st.error("API 응답의 'list' 항목이 비어 있습니다. 반환된 데이터:" + str(data))
        return pd.DataFrame()
    df = pd.DataFrame(financial_list)
    if "account_nm" in df.columns:
        df["account_nm"] = df["account_nm"].str.strip()
        st.write("가져온 계정명들:", df["account_nm"].unique())
    else:
        st.write("account_nm 열이 데이터프레임에 없습니다. df.columns:", df.columns)
    return df

def extract_financial_items(df):
    mapping = {
        "Total_Assets": "자산총계",
        "Total_Liabilities": "부채총계",
        "Total_Equity": "자본총계",
        "Sales": "매출액",
        "Operating_Profit": "영업이익",
        "Net_Income": "당기순이익",
        "Debt_Ratio": "부채비율",
        "Sales_Growth": "매출액증가율",
        "Operating_Profit_Growth": "영업이익증가율",
        "Net_Income_Growth": "당기순이익증가율",
        "Sales_Status": "매출액 상태",
        "Operating_Profit_Status": "영업이익 상태",
        "Net_Income_Status": "당기순이익 상태",
        "ROE": "ROE",
        "ROA": "ROA",
        "Free_Cash_Flow": "자유현금흐름",
        "Dividend_Yield": "배당수익률"
    }
    items = {}
    for key, account in mapping.items():
        try:
            value = df.loc[df['account_nm'] == account, 'thstrm_amount'].iloc[0]
            if key in ["Sales_Status", "Operating_Profit_Status", "Net_Income_Status"]:
                items[key] = value
            elif key in ["Debt_Ratio", "Sales_Growth", "Operating_Profit_Growth", "Net_Income_Growth", "ROE", "ROA", "Dividend_Yield"]:
                items[key] = float(value.replace(",", "")) / 100
            else:
                items[key] = float(value.replace(",", ""))
        except Exception as e:
            st.write(f"계정명('{account}') 데이터 추출 실패: {e}")
            items[key] = None

    # 누락된 필드는 기본값 적용
    items.setdefault("PER", 15.0)
    items.setdefault("PBR", 1.5)
    defaults = {
        "Free_Cash_Flow": 0,
        "Dividend_Yield": 0.02,
        "ROE": 0.12,
        "ROA": 0.04,
        "Debt_Ratio": 0.70,
        "Sales_Growth": 0,
        "Operating_Profit_Growth": 0,
        "Net_Income_Growth": 0,
        "Sales_Status": "보통",
        "Operating_Profit_Status": "보통",
        "Net_Income_Status": "보통"
    }
    for key, val in defaults.items():
        if items.get(key) is None:
            items[key] = val
    return items

# LLM에 계산과 이유 설명을 요청하는 프롬프트 템플릿
prompt_template = """
당신은 금융 분석 전문가입니다. 아래 재무 지표를 바탕으로 장기 투자 가치를 평가합니다.
점수는 초기 30점에서 시작하여 아래 기준에 따라 가감합니다.

[기준]
- PER: 10 미만이면 -4, 15 미만이면 -2, 20 미만이면 0, 그 이상이면 +4.
- PBR: 1.0 미만이면 -4, 1.5 미만이면 -2, 2.0 미만이면 +1, 그 이상이면 +3.
- ROE: 0.20 이상이면 -4, 0.15 이상이면 -2, 0.10 이상이면 +1, 그 미만이면 +4.
- ROA: 0.07 이상이면 -3, 0.05 이상이면 -1, 0.03 이상이면 +1, 그 미만이면 +3.
- 부채비율: 0.50 미만이면 -3, 0.70 미만이면 -1, 1.0 미만이면 +1, 그 이상이면 +4.
- 매출액, 영업이익, 당기순이익 증가율: 각 지표가 10% 이상이면 -2, 음수면 +3 (각 항목마다 적용).
- 매출액, 영업이익, 당기순이익 상태: "양호" 포함 시 -1, "부진" 포함 시 +2, 그 외에는 0.
- 자유현금흐름: 양수이면 -2, 음수이면 +3.
- 배당수익률: 0.04 이상이면 -2, 0.02 이상이면 -1, 그 미만이면 +2.

모든 항목에 대한 가중치 계산 후 최종 점수는 0 이상 59 이하로 보정합니다.
또한, 최종 점수를 아래 규칙에 따라 등급으로 매깁니다.
- 점수를 10으로 나눈 몫을 글자로 변환합니다: {{0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F"}}
- 나머지에 1을 더한 숫자와 결합합니다.
예를 들어, 점수가 32이면 32 // 10 = 3 → "D", 그리고 32 % 10 + 1 = 3 → "D3".

아래 재무 지표를 참고하여 최종 점수와 등급을 계산하고, 각 항목이 최종 점수에 어떤 영향을 미쳤는지 상세하게 설명한 후,
JSON 형식으로 출력하세요. 출력 예시는 다음과 같이 해주세요:

{{
  "final_score": <숫자>,
  "grade": "<등급>",
  "explanation": "<각 항목에 대한 가감 설명>"
}}

[재무 지표]
PER: {PER}
PBR: {PBR}
ROE: {ROE}
ROA: {ROA}
Debt_Ratio: {Debt_Ratio}
Sales_Growth: {Sales_Growth}
Operating_Profit_Growth: {Operating_Profit_Growth}
Net_Income_Growth: {Net_Income_Growth}
Sales_Status: {Sales_Status}
Operating_Profit_Status: {Operating_Profit_Status}
Net_Income_Status: {Net_Income_Status}
Free_Cash_Flow: {Free_Cash_Flow}
Dividend_Yield: {Dividend_Yield}
"""

prompt = PromptTemplate(
    input_variables=[
        "PER", "PBR", "ROE", "ROA", "Debt_Ratio",
        "Sales_Growth", "Operating_Profit_Growth", "Net_Income_Growth",
        "Sales_Status", "Operating_Profit_Status", "Net_Income_Status",
        "Free_Cash_Flow", "Dividend_Yield"
    ],
    template=prompt_template,
)

llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)

st.title("국내주식 투자가치 평가 (장기 투자) - LLM 활용")

user_input = st.text_input("분석할 국내 주식의 기업명을 입력하세요 (예: 삼성전자):")
bsns_year = st.text_input("사업년도 (예: 2022):", "2022")
reprt_code = st.text_input("보고서 코드 (예: 11011):", "11011")

if st.button("분석하기") and user_input:
    try:
        stock_code = get_stock_code(user_input)
        df = fetch_dart_financials_df(stock_code, bsns_year, reprt_code)
        if df.empty:
            raise Exception("재무 데이터가 없습니다. (DataFrame 결과가 비어 있습니다.)")
        dart_financials = extract_financial_items(df)
        st.write("가공된 재무 지표:", dart_financials)
        
        # LangChain을 이용해 LLM에게 최종 점수와 이유를 요청
        llm_response = chain.run(**dart_financials)
        st.subheader("LLM 투자 평가 결과")
        st.markdown("최종 평가 결과 (LLM 기반):")
        st.write(llm_response)
    except Exception as e:
        st.error("오류 발생: " + str(e))
