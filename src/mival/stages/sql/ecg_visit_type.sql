-- The visit type of every ECG of the modality: the standard concept name of
-- the visit its image_occurrence names. An ECG with no visit_occurrence_id, or
-- one that names a visit not in the table, comes back with a null name.
-- {schema} is filled by the stage after an identifier check.
select
    i.image_occurrence_id,
    v.visit_concept_id,
    c.concept_name                                   as visit_concept_name
from {schema}.image_occurrence i
left join {schema}.visit_occurrence v on v.visit_occurrence_id = i.visit_occurrence_id
left join {schema}.concept c on c.concept_id = v.visit_concept_id
where i.modality_concept_id = %(modality_concept_id)s
