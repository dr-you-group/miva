-- 실행 전 코호트 규모 확인용 (SQL Server). PostgreSQL은 check_cohort.postgresql.sql. 파이프라인은 이 파일을 쓰지 않는다.
-- cdm 을 기관 schema(예: CDM2025.dbo) 로, 4145308 을 기관의 12-lead ECG modality concept 으로 바꿔서 실행한다.
-- 결과의 숫자는 기관 밖으로 내보내지 않는다.

-- 1. ECG 전체와 local_path 예시 (파이프라인은 local_path를 적힌 그대로 읽는다. example_path가 이 서버에서 열려야 한다)
select count(*)                                          as n_ecg,
       count(distinct person_id)                         as n_person,
       min(local_path)                                   as example_path
from cdm.image_occurrence
where modality_concept_id = 4145308;

-- 2. LVEF (3027172, %) 가 ECG 날짜 ±7일 안에 있는 ECG
select count(distinct i.image_occurrence_id) as n_ecg_with_lvef
from cdm.image_occurrence i
join cdm.measurement m
  on m.person_id = i.person_id
 and m.measurement_concept_id = 3027172
 and m.value_as_number is not null
 and abs(datediff(day, i.image_occurrence_date, m.measurement_date)) <= 7
where i.modality_concept_id = 4145308;

-- 3. 혈청 칼륨 (3023103, mmol/L) 이 같은 날 있는 ECG
select count(distinct i.image_occurrence_id) as n_ecg_with_potassium,
       min(m.value_as_number) as k_min, max(m.value_as_number) as k_max
from cdm.image_occurrence i
join cdm.measurement m
  on m.person_id = i.person_id
 and m.measurement_concept_id = 3023103
 and m.value_as_number is not null
 and m.measurement_date = i.image_occurrence_date
where i.modality_concept_id = 4145308;

-- 4. 나이·1년 사망에 필요한 person / death / visit_occurrence
select count(distinct i.person_id)                                       as n_person,
       sum(case when p.year_of_birth is null then 1 else 0 end)          as n_ecg_no_birth_year,
       count(distinct d.person_id)                                       as n_person_with_death
from cdm.image_occurrence i
join cdm.person p on p.person_id = i.person_id
left join cdm.death d on d.person_id = i.person_id
where i.modality_concept_id = 4145308;
