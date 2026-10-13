-- Every ECG with the person attributes the derived label kinds need: year of
-- birth, death date, and the end of the person's last visit (which bounds how
-- long an out-of-hospital death could still have been recorded). {schema} is
-- filled by the stage after an identifier check.
select
    i.image_occurrence_id,
    i.person_id,
    i.local_path,
    i.image_occurrence_date                          as index_datetime,
    p.year_of_birth,
    p.gender_concept_id,
    d.death_date,
    v.last_visit_end
from {schema}.image_occurrence i
join {schema}.person p on p.person_id = i.person_id
left join (
    select person_id, min(death_date) as death_date from {schema}.death group by person_id
) d on d.person_id = i.person_id
left join (
    select person_id, max(visit_end_date) as last_visit_end from {schema}.visit_occurrence group by person_id
) v on v.person_id = i.person_id
where i.modality_concept_id = %(modality_concept_id)s
order by i.image_occurrence_id
