# #10 인공지능 기반 CCTV 영상분석을 통한 도시침수 감지

분석 주제 및 목표: CCTV영상에서 침수 영역을 추출하는 물 객체 감지 및 영역 추출 모델 생성

Detectron2의 Mask R-CNN을 이용한 Instance Segmentation

핵심 아이디어: Instance Segmentation을 이용해 객체의 위치를 실제 edge에 대한 정보를 바탕으로 추론

1. 데이터 셋 준비
2. 모델 생성 및 학습
3. 예측 및 결과
4. 물 영역 면적 추출
5. 영상 분석

1. 데이터 셋 준비
    
    (1) labelme를 이용해 이미지의 물 객체에 대해 폴리곤으로 영역을 지정하고 물 객체 좌표와 클래스 등을 담은 json파일을 얻을 수 있는데, 이를 Detectron2에서 사용하는 데이터 방식(COCO 데이터 형식)으로 바꿔줄 필요가 있음
    
    - get_data_dicts(가져올 json파일이 존재하는 디렉토리, class이름이 들어가 있는 list)
        - json 파일을 가져오고 이미지 경로를 참조해서 jpg파일도  가져올 수 있음
            - record라는 딕셔너리에 파일이름, 이미지 높이,너비, 이미지 id(커스텀 변수, 매 시행마다 +1) 추가
            - annos라는 리스트에 현재 이미지의 객체 정보(좌표 값들, label한 클래스 명)를 가져온 json파일의 shapes라는 key값을 이용해서 참조하고 저장
            - 객체 정보를 바탕으로 labelme에서 폴리곤으로 지정한 영역의 좌표값인 x값들과 y값들을 얻어와 얻어와 obj라는 딕셔너리에 저장
                - bbox: bounding box 좌표값 4개
                - bbox_mode: bbox 포맷
                - segmentation: x값과 y값을 묶은 튜플들의 list의 list
                - category_id: 본 함수에서 매개변수로 사용한 class이름이 들어가 있는 list에서 현재 이미지  객체의 label이 몇 번째 인덱스인지
                - iscrowd: 추론하려는 해당 물체가 밀집 영역에 속해있는지(모든 물체가 분리되어 있어 0)
            - objs라는 리스트에 obj딕셔너리들을 저장
            - 이미지에 대한 정보를 저장하는 record 딕셔너리의 “annotation” key에 대한 value 값으로 objs 리스트를 저장
            - 모든 이미지들에 대한 정보를 저장하는 리스트인 dataset_dicts에 record 딕셔너리를 추가
        - dataset_dicts 리스트를 return
    
    (2) Detectron2 모델의 데이터셋에 현재 가지고 있는 데이터들이 존재하지 않기 때문에  추가해주는 과정이 필요함
    
    - DatasetCatalog.register(str param1, list param2)
        - param1: Detectron2에서 사용할 데이터셋의 이름
        - param2: train과 test데이터들이 존재하는 디렉토리에 대해  get_data_dicts 함수를 적용하여 Detectron에서 사용하는 데이터 형태인 COCO 데이터 셋으로 변환
        
        ```python
        DatasetCatalog.register('등록할 데이터셋 이름',등록할 데이터)
        ex)DatasetCatalog.register('category_' + d, lambda d = d: get_data_dicts(data_path + d, classes))  # DatasetCatalog에 category_train, category_test 등록
        ```
        
    - MetadataCatalog.get(string param1).set(thing_class = param2)
        - param1: Detectron2의 데이터셋에 접근하여 데이터셋 이름, 해당 과정을 통해 메타데이터 정보를 가져올 수 있음
        - param2: 접근한 해당 데이터셋에 메타데이터 정보를 설정. get_data_dicts에서 사용한 클래스 리스트를 사용해 해당 데이터셋의 클래스 레이블을 설정
        
        ```python
        MetadataCatalog.get('가져올 데이터셋 이름').set(thing_classes = 등록하려는 클래스 배열)
        MetadataCatalog.get('category_' + d).set(thing_classes = classes)  # MetadataCatalog에 category_train, category_test 등록
        ```
        
2. 모델 생성 및 학습
    
    <img src = '/images/Mask_R-CNN_구조.png' alt = 'Drawing' style = 'width: 600px;'/>
    
    (이미지 출처) https://<hi>[techblog-history-younghunjo1.tistory.com/193](http://techblog-history-younghunjo1.tistory.com/193)
    
    (1) COCO 데이터셋으로 사전 학습된 Mask-R-CNN 모델을 로드하고 현재 가지고 있는 데이터들로 fine-tuning을 수행해 물 객체 감지 및 영역 추출 모델을 생성
    
    - RPN: 이미지에서 RoI(객체를 찾기 위한 후보영역) 생성
    - RoI Align: RoI를 Feature Map으로 변환(이미지에서 추출된 특징을 나타내는 2D 배열)
    - 픽셀 수준의 마스크 분할을 수행하는 FCN에 RoI를 전달함
        - FCN은 RoI를 입력으로 받아 RoI내의 픽셀들에 대한 분류와 마스크 분할을 수행함(물체의 윤곽 추출 가능)
    - Detectron2의 가중치를 가져오고 모델 아키텍처를 구현하기 위해서 Detectron2의 config파일을 가져와 모델 아키텍처, 데이터셋 경로, 학습 파라미터 설정
    
    (2) 모델 및 학습에 필요한 하이퍼파라미터 설정
    
    - get_cfg: Detectron2에서 사용되는 설정 파일을 가져와 Node를 생성
    - merge_from_file(model_zoo.get_config_file(”COCO 인스턴스 세그멘테이션을 위한 파일”))
        - model_zoo: 사전 학습된 모델들을 모아놓은 저장소
        - get_config_file(”설정 파일”): 설정 파일의 경로를 반환
        - 설정 파일에 저장된 설정을 현재 설정에 병합
    - DATASETS.TRAIN: Detectron2의 데이터셋에 존재하는 train_data의 데이터셋 이름을 tuple 형태로 지정
    - DATALOADER.NUM_WORKERS
        - 데이터로더가 사용하는 CPU 코어의 수 제어
    - MODEL.WEIGHTS
        - get_checkpoint_url(’설정 yaml 파일 디렉토리’)사전 학습된 모델 MASK R-CNN 모델의 가중치를 가져옴
    - SOLVER.IMS_PER_BATCH
        - 모델 학습 중 사용할 이미지 배치 크기로, 더 큰 배치는 더 빠른 학습을 가능하게 하지만  GPU 메모리에 맞게 설정해야함
    - SOLVER.BASE_LR
        - 모델의 가중치를 업데이트하는 속도인 학습률을 설정하는 하이퍼파라미터로 BASE_LR을 통해 초기 학습률을 지정
    - SOLVER.MAX_ITER
        - 모델이 학습을 언제 중지해야 하는지를 지정하는 값
        - 모델 학습의 최대 반복 횟수: 모델이 데이터 셋을 몇 번 반복하고 가중치를 업데이트 할 지를 지정
        - 값이 크면 학습 데이터를 더 잘 이해하고 일반화 할 수 있는 가능성이 높아지지만 과적합되어 검증 데이터에서 나쁜 성능을 보일 가능성 높음(*)
        - 값이 작다면 모델이 학습 데이터에 잘 적합되지 못하는 underfitting이 발생하고 일반화 능력을 감소시켜 작은 데이터셋에서 과적합이 발생할 수 있음
    - MODEL.ROI_HEADS.NUM_CLASSES
        - 모델이 분류해야 하는 클래스 개수. 본 과제에서는 water class의  탐지만을 목표로 하므로 1
    
    (3) 모델 학습
    
    - 모델 학습 후 결과물을 저장할 디렉토리 설정
        - cfg.OUTPUTDIR
            - cfg: Detectron2의 설정 파일을 가져와 생성한 Node
            - 모델 학습 결과 및 로그 파일을 저장
            - 체크포인트 파일, 학습 곡선 그래프 파일, 로그 파일 저장
    - DefaultTrainer(CfgNode)
        - Detectron2 라이브러리에서 딥러닝 모델 학습을 위한 훈련 루프를 구성하고 관리하는 클래스
        - Detectron2 설정파일(cfg)를 매개변수로 받아 모델, 데이터로더, 학습옵션 등을 정의
    - trainer.resume_or_load(resume=)
        - 이전에 학습된 가중치를 불러와 이어서 학습할지(true) 처음부터 학습할지(false)결정
        - 사전 학습된 모델을 여전히 사용하면서 새로운 데이터셋에 맞게 일부 레이어를 초기화하고 처음부터 다시 학습
    - trainer.train()
        - events.out.tfevents: TensorBoard를 위한 이벤트 파일
        - model_final_pth: 최종 학습된 모델의 가중치
        - metric.json: 학습 중 측정된 평가 지표
        - last_checkpoint: 마지막으로 저장된 체크포인트 파일의 경로
    - metric.json
        - total_loss(손실 함수에서 계산된 모든 손실값들의 총합)을 이용해 Loss curve가 점차 감소하고 있는 지 확인
        - fast_rcnn/cls_accuracy(모델이 정확하게 객체를 인식한 비율)을 이용해 1에 가까운 큰 값이 나오는 지 확인
    
    1. 예측 및 결과
    - Detectron2의 설정 파일로 생성한 노드(CfgNode)의 가중치를 학습 후 저장된 모델의 가중치 경로로 지정
        
        ```python
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
        ```
        
    - 모델의 예측에 대한 스코어 임계값 설정
    - 모델 평가에 사용될 데이터 셋 적용: cfg.DATASETS.TEST
        
        ```python
        cfg.DATASETS.TEST = ('category_test', )
        ```
        
    - DefaultPredictor(cfg): test 데이터 셋에 대해 객체 검출 예측을 수행하고 결과를 출력
    
    1. 물 영역 면적 추출
        - 물 영역을 흑백으로 표시하는 이진화 작업 수행
        - 범용 모폴로지 연산을 이용해 닫기 연산을 수행
        - 외곽선을 찾고 꼭짓점 좌표만 반환
        - 외곽선을 이미지에 그리기
        - 외곽선을 통해 면적 구하기
        - 면적 비율값에 따라 경보 단계를 표시
        - 이미지에 텍스트를 작성
    
    1. 영상 분석
        - 유튜브 다운로드를 통한 동영상 파일을 가져오고 10프레임마다 이미지를 캡처해 저장
        - Instance Segmentation 수행을 위해 이미지들에 Detectron2의 Visualizer함수를  사용
            - BGR2RGB
            - MetadataCatalog를 이용해 학습된 모델이 사용된 데이터셋의 메타데이터를 가져옴
        - Detectron2의 Visualizer 클래스의 메서드  중 하나인 draw_instance_predictions를 이용하여 객체 감지 모델의 결과를 시각적으로 나타낼 수 있는 이미지로 반환
        - 분할 결과 중 물 영역만 추출하여 흰색으로 표시, 배경은 검은색
        - 이미지를 이진화 후 범용 모폴로지 연산 적용
        - 입력 이미지에서 cv2의 findContours 함수를 이용해 외곽선을 찾기
        - cv2의 drawContours 함수를 이용해 외곽선을 이미지에 그리기
        - 외곽선을 통해 면적을 구하고 면적 퍼센티지에 따라 경보 단계를 표시
