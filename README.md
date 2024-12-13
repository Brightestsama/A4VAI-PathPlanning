# 통합측 변경 내용 

## custom_msgs, os import 추가
	- custom_msgs에는 공통 Interface topic massage 정의 

## Image2Plan.py 삭제 
	- 이미지 경로 , 모델 경로 하드코딩
	- GlobalWaypointSetpoint은 Controller에서 Publish
	- Mode는 컨테이너 환경변수로 대체

## def compute_path 함수 z 고도값 150에서 5로 변경 

## Global Waypoint 및 Local waypoint
	- main 함수에서 PathPlanningServer Node 내부로 path planning 알고리즘 이동
		- global_waypoint_callback에 정의
	- Controller로부터 GlobalWaypointSetpoint(x,y,z) subscribe
	- subscribe 하면 path planning 실행 및 local waypoint publish

	- global_waypoint_subscriber 추가 
	- global_waypoint_callback 추가
	- local_waypoint_publisher 추가

## heartbeat check 
	- 1초에 한번씩 heartbeat publish
	- 모든 모듈 heartbeat True 일시 global_waypoint_callback 작동 

	- heartbeat check function 추가
	- heartbeat_publisher 추가 
	- heartbeat_subscriber 추가 
	- heartbeat flag 추가
	- heartbeat_timer 추가


## 수정 필요 사항 
	- 이미지 경로 현재 하드코딩 -> Controller에서 topic으로 보내주도록 변경
