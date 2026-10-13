-- The label table as it stood when this run read it (spec §4.1). A cohort is
-- only reproducible against the snapshot it was drawn from, and these two
-- numbers say whether the source table has grown since. {schema} is filled by
-- the stage after an identifier check; the concept id is a bound parameter.
select
    count(*)            as n_rows,
    max(measurement_id) as max_measurement_id
from {schema}.measurement
where measurement_concept_id = %(label_concept_id)s
