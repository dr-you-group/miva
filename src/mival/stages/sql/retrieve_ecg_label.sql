-- Every ECG with the nearest label within the window, or nulls when none.
-- Ties on day distance go to the earlier measurement. {schema} is filled by
-- the stage after an identifier check; the other values are bound parameters.
-- The between clause states the same window as the abs() one in a form the
-- (person_id, measurement_date) index can range-scan.
select
    i.image_occurrence_id,
    i.person_id,
    i.local_path,
    i.image_occurrence_date                          as index_datetime,
    m.measurement_datetime                           as label_datetime,
    m.measurement_date - i.image_occurrence_date     as label_delta_days,
    m.value_as_number                                as label_value
from {schema}.image_occurrence i
left join lateral (
    select m.measurement_datetime, m.measurement_date, m.value_as_number
    from {schema}.measurement m
    where m.person_id = i.person_id
      and m.measurement_concept_id = %(label_concept_id)s
      and m.value_as_number is not null
      and m.measurement_date between i.image_occurrence_date - %(window_days)s
                                 and i.image_occurrence_date + %(window_days)s
      and abs(m.measurement_date - i.image_occurrence_date) <= %(window_days)s
    order by abs(m.measurement_date - i.image_occurrence_date), m.measurement_datetime
    limit 1
) m on true
where i.modality_concept_id = %(modality_concept_id)s
order by i.image_occurrence_id
