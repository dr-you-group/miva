-- SQL Server form of retrieve_ecg_label.sql: the same rows, with outer apply
-- for the lateral join and datediff/dateadd for the date arithmetic.
-- Every ECG with the nearest label within the window, or nulls when none.
-- Ties on day distance go to the earlier measurement. {schema} is filled by
-- the stage after an identifier check; the other values are bound parameters.
-- The between clause states the same window as the abs() one in a form the
-- (person_id, measurement_date) index can range-scan.
select
    i.image_occurrence_id,
    i.person_id,
    i.local_path,
    i.image_occurrence_date                                          as index_datetime,
    m.measurement_datetime                                           as label_datetime,
    datediff(day, i.image_occurrence_date, m.measurement_date)      as label_delta_days,
    m.value_as_number                                                as label_value
from {schema}.image_occurrence i
outer apply (
    select top 1 m.measurement_datetime, m.measurement_date, m.value_as_number
    from {schema}.measurement m
    where m.person_id = i.person_id
      and m.measurement_concept_id = %(label_concept_id)s
      and m.value_as_number is not null
      and m.measurement_date between dateadd(day, -%(window_days)s, i.image_occurrence_date)
                                 and dateadd(day, %(window_days)s, i.image_occurrence_date)
      and abs(datediff(day, i.image_occurrence_date, m.measurement_date)) <= %(window_days)s
    order by abs(datediff(day, i.image_occurrence_date, m.measurement_date)), m.measurement_datetime
) m
where i.modality_concept_id = %(modality_concept_id)s
order by i.image_occurrence_id
