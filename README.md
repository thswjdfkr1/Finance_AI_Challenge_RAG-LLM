# Finance_AI_Challenge
금융 및 금융보안 관련 문서를 바탕으로 주어진 객관식/주관식 문제에 대한 답을 자동으로 생성하는 RAG 기반 금융 보안 도메인 특화 sLLM 구축

# 주제
금융보안 실무에 적합한 개발 방법론 및 AI 모델을 발굴하기 위해 FSKU 평가지표를 기반으로 AI 모델의 성능을 경쟁

# 사용 기술
- Python
- HuggingFace
- Open LLM(LGAI-EXAONE, SKT-A.X)
- Transformer
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
  


여기서부터 하면 돼   



# RAG      
1. 로드              
   - PDF 문서 로딩을 위해 PyPdfReader를 사용하여 문서 처리
2. 문서 정리 및 클린징           
3. Split        
   - RecursiveCharacterTextSplitter를 사용하여 문서를 잘게 분할 
4. Embedding    
   - 텍스트 임베딩에는 SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 모델 사용   
   - 선택이유
      * 이 모델은 빠르고 효율적으로 텍스트를 벡터 형태로 변환하여 의미 기반 검색에 적합한 표현을 제공   
      * 이 과정을 통해 각 문서가 의미적으로 잘 표현된 벡터로 변환되어, 검색 및 후속 처리에서 높은 성능을 발휘 
5. Store
   - 문서를 하이브리드 검색 방식을 사용하기 위한 형식으로 저장       
     

# 추론    
### LLM 파인튜닝     
1. 모델 및 토크나이저 설정      
* model = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'     

2. LoRA 경량화
   
   LoRA (Low Rank Adaptation)는 파인튜닝을 위한 경량화 기법     
   pre-trained 모델에 가중치를 고정하고, 각 계층에 훈련 가능한 랭크 분해 행렬을 주입하여 훈련 가능한 매개 변수의 수를 크게 줄일 수 있음.      
   LoRA를 사용하면 기존 모델의 대규모 파라미터를 전부 재학습할 필요 없이, 소수의 추가 파라미터만을 학습하여 모델을 새로운 태스크에 적응시킬 수 있어, 전체 모델을 처음부터 다시 학습하는 것보다 훨씬 적은 계산 자원을 사용하여, 시간과 비용을 절
   약 할 수 있음    

3. Trainning    

   경량화를 마친 모델에 QADataset을 학습    

4. Model Load    

   ```
   adapter_path = "/content/drive/MyDrive/1데이콘/2025금융AIChallenge금융AI모델경쟁/dataset/finetunning_model8/checkpoint-1104"   
   fine_model = PeftModelForCausalLM.from_pretrained(model, adapter_path)    
   fine_model = fine_model.merge_and_unload().to("cuda")
   ```

  merge_and_unload을 통해 Model 로드 후 학습된 adapter을 결합 후 제거     

### 추론     
1. 하이브리드 검색기     
* BM25/FAISS 임베딩
* Top-K 문서 선택
```
  top_indices = np.argsort(combined_scores)[::-1][:top_k]
  top_docs = [all_chunks[i] for i in top_indices]

  return top_docs, combined_scores
```

2. Prompt
``` 
def make_prompt_auto(text: str, top_docs: str) -> str:
    """RAG 컨텍스트를 포함해 객관식/주관식 프롬프트를 자동 구성"""
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요. 다른 단어/설명 금지.\n\n"
            "예: 1 / 2/ 3/ 4/ 5\n\n"
            f"참고문서: {top_docs}\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{'\n'.join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
            "단, 참고 문서를 바탕으로 답을 구성하되 검색된 내용을 그대로 복사하지 말고 반드시 **재구성, 요약, 재작성**해서 답변해야 합니다.\n\n"
            f"참고문서: {top_docs}\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )
    return prompt
```

2. 대책 생성 함수 (generate_prevention_plan)
* 객관식 / 주관식에 따른 답변 생성
```
  # 객관식
  if is_multiple_choice(question):
    output_ids = fine_model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )
  else:
    output_ids = fine_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
      )

  output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  pred_answer = extract_answer_only(output_text, original_question=q)

  return pred_answer
```

3. 추론

### 설명   
> text: 금융 관련 질문, top_docs: 하이브리드 검색을 통해 검색된 관련 문서

이를 기반으로 "금융 관련 질문: {text} \n 관련 문서: {top_docs}" 형식으로 구성 토큰화 및 모델 입력 준비

### PEFT 모델을 활용한 문장 생성
* bm25, faiss와 각각의 가중치를 설정하여 top_docs 문서 추출    
* fine_model.generate(**inputs, max_length=256) 주관식 문제의 경우 최대 256자 길이로 답을 생성     
* tokenizer.decode(output_ids[0], skip_special_tokens=True) 특수 토큰을 제거하고 최종 답안을 반환     
   
# 성과 :  
- 답변 정확도 63% 달성  
-	검색-추론-생성 결합 RAG 구조 적용으로 추론 정확도 10% 향상  
-	BaseLine 대비 전체 정확도 약 53% 이상 개선 달성
  
