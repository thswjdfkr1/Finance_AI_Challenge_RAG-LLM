# Finance_AI_Challenge
금융 및 금융보안 관련 문서를 바탕으로 주어진 객관식/주관식 문제에 대한 답을 자동으로 생성하는 RAG 기반 금융 보안 도메인 특화 sLLM 구축

# 주제
금융보안 실무에 적합한 개발 방법론 및 AI 모델을 발굴하기 위해 FSKU 평가지표를 기반으로 AI 모델의 성능을 경쟁

# 사용 기술
- Python
- HuggingFace Transformers
- Langchain
- Open LLM(LGAI-EXAONE, SKT-A.X)
- Fine-Tunning
- PEFT
- LoRA
- BM25, FAISS(Retriever)
- Prompt_Engineering     


# QA Dataset 생성		   
1. 로드              
   - PDF 문서 로딩을 위해 PyPdfReader를 사용하여 문서 처리
2. 문서 정리 및 클린징           
3. Split        
   - RecursiveCharacterTextSplitter를 사용하여 문서를 잘게 분할    
4. QADataset셋 생성     
   - 생성된 chunk를 모두 합쳐 'skt/A.X-4.0-Light' 오픈 모델을 활용하여 zero-shot 기반 QADataset 생성      
   
# LLM 파인튜닝     
1. 모델 및 토크나이저 설정
   - model = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'
2. 양자화
   - BitsAndBytes 기반 4bit 양자화 적용(가중치를 4bit로 줄임)
3. 경량화
   - rank = 16
   - alpha = 32
   - dropout = 0.1
   - target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
* 전체 파라미터 중 0.1246% 만 학습 
4. Trainning
   - 양자화 및 경량화를 거친 모델에 기존 생성한 QADataset 학습    
5. 모델 결합    
   - merge_and_unload을 통해 경량화 및 양자화를 거친 adapter를 based 모델과 결합
  
# RAG      
1. 로드              
   - PDF 문서 로딩을 위해 PyPdfReader를 사용하여 문서 처리
2. 문서 정리 및 클린징           
3. Split        
   - RecursiveCharacterTextSplitter를 사용하여 문서를 잘게 분할 
4. Retriever
   (1) FAISS    
      - 텍스트 임베딩에는 SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 모델 사용   
      -  MMR(Maxiaml Marginal Relevance) 방식 적용
         * k=5
         * fetch_k=20
         * lambda_multi=0.5
      - 유사도 + 다양성 동시에 고려하여 탐색
      - 벡터 매칭 기반 점수 산
   (2) BM25
      - BM25Okapi 사용
      - 문서를 토큰 단위 분리 후 sparse 검색기 구축
      - 키워드 매칭 기반 점수 산출    
5. Hybrid Search
   - BM25 점수 -> Min-Max 정규화
   - FAISS 거리값 -> 유사도 점수로 변환 후 정규화
   - 최종 점수 -> w1 * BM25_scores + w2 * FAISS_scores
   - 상위 Top-K 문서 선택
   * Dense + Sparse 결합으로 검색 안전성 강화

# 추론       
1. top-k 문서 추출
2. prompt 적용
3. 객관식/주관식 여부 판단 함수 적용
4. 답변 추론
5. 후처리 함수 적용
6. 최종 답변 추론
   
# 성과 :  
- 답변 정확도 63% 달성  
-	검색-추론-생성 결합 RAG 구조 적용으로 추론 정확도 10% 향상  
-	BaseLine 대비 전체 정확도 약 53% 이상 개선 달성
  
