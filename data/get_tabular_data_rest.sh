#!/bin/bash

python ../src/eds/webapi_test_clients/webapi_client_rest.py trend_tabular --host="https://eds-demo.tt.com.pl/webapi" --user=multidemo --password=demo_pass from=2025-09-04T00:00:00 till=2025-09-08T00:00:00 step=3 iess=watt_hi.common@irr_1 iess=watt_hi.common@irr_2 iess=watt_hi.common@irr_3 iess=watt.common@irr_dav_1 function=MAX_VALUE
#> ./org/25-09-04_08_rest.csv
