-- One acquisition parameter of every ECG of the modality: the image_feature
-- rows of one concept whose event is a measurement row (field concept 1147330,
-- measurement.measurement_id), with that measurement's numeric value. A
-- parameter stored once per channel comes back once per channel; the stage
-- reduces it to one value per ECG. {schema} is filled by the stage after an
-- identifier check.
select
    f.image_occurrence_id,
    m.value_as_number
from {schema}.image_feature f
join {schema}.image_occurrence i on i.image_occurrence_id = f.image_occurrence_id
join {schema}.measurement m on m.measurement_id = f.image_feature_event_id
where i.modality_concept_id = %(modality_concept_id)s
  and f.image_feature_concept_id = %(feature_concept_id)s
  and f.image_feature_event_field_concept_id = 1147330
  and m.value_as_number is not null
