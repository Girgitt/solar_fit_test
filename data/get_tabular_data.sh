#!/bin/bash

python ../../src/eds/webapi_test_clients/webapi_client.py getTabularCSV wsdlUrl="https://eds-demo.tt.com.pl/webapi/eds.wsdl" user=multidemo password=demo_pass PointId=test from=2025-09-04T00:00:00 till=2025-09-08T00:00:00 step=5 point=watt_hi.common@irr_1 point=watt_hi.common@irr_2 point=watt_hi.common@irr_3 point=watt.common@irr_dav_1 function=MAX_VALUE > ./org/25-09-04_08.csv
