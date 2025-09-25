#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Niniejsze dane stanowią tajemnicę przedsiębiorstwa.
# The contents of this file is proprietary information.

import sys

import math
import time
import json
import io
import re
import argparse
import requests
from collections import OrderedDict

parser = argparse.ArgumentParser(description='WebAPI REST client')
parser.add_argument('endpoint', help='endpoint')
parser.add_argument('--method', help='HTTP method', default='POST')
parser.add_argument('--user', help='user name', default='admin')
parser.add_argument('--password', help='password', default='')
parser.add_argument('--sessionid', help='session ID')
parser.add_argument('--host', help='host URL', default='http://127.0.0.1:43084')
parser.add_argument('--verify_cert', help='verify SSL certificate', action='store_true')

group = parser.add_argument_group('cram tests utils')
group.add_argument(
    '--expectedMatchCount', type=int,
    help='Expected value of "matchCount" field in response')
group.add_argument(
    '--noRequestExecutionLogs',
    action='store_true',
    help="Do not print additional logs from query execution")
group.add_argument(
    '--floatPrecision', type=int, default=None)

class Client:

    def __init__(self, args, params):
        self.args = args
        self.params = params
        self.api = args.host + '/api/v1/'
        self.maxRepeats = 5
        self.repeatTimeBreak = 1
        self.verify = args.verify_cert
        self.session = requests.Session()
        if args.sessionid:
            self.session.headers['Authorization'] = 'Bearer ' + args.sessionid
        self._origParams = params.copy()
        self._noRequestExecutionLogs = args.noRequestExecutionLogs
        self._floatPrecision = args.floatPrecision

    def run(self, endpoint, verbose=True):
        reapetCounter = 0
        while reapetCounter < self.maxRepeats:
            reapetCounter += 1
            try:
                if endpoint.startswith('/'):
                    print('GET ' + self.args.host + endpoint)
                    return self.session.get(self.args.host + endpoint)

                method = getattr(self, endpoint, None)
                if method is None:
                    print('Invalid method "{}"'.format(endpoint))
                    exit(-1)
                res = method()
                if res is not None:
                    if not is_valid_response(res) or not self._is_expected_result(res):
                        time.sleep(self.repeatTimeBreak)
                        self.params = self._origParams.copy()
                        continue

                    log_request(res, verbose=verbose)
                    break
                else:
                    break
            except:
                time.sleep(self.repeatTimeBreak)
                self.params = self._origParams.copy()

    def authenticate(self):
        data = OrderedDict()
        data['username'] = self.args.user
        data['password'] = self.args.password
        return self.session.post(self.api + 'authenticate', json=data, verify=self.verify)

    def login(self):
        data = OrderedDict()
        data['username'] = self.args.user
        data['password'] = self.args.password
        data['type'] = 'myapp'
        res = self.session.post(self.api + 'login', json=data, verify=self.verify)
        if res.status_code == 200:
            self.session.headers['Authorization'] = 'Bearer ' + res.json()['sessionId']
        return res

    def logout(self):
        return self.session.post(self.api + 'logout', verify=self.verify)

    def ping(self):
        return self.session.get(self.api + 'ping', verify=self.verify)

    def objects(self):
        if self.args.method == 'POST':
            data = [self.params]
            return self.session.post(self.api + 'objects', json=data, verify=self.verify)
        elif self.args.method == 'PUT':
            data = [self.params]
            return self.session.put(self.api + 'objects', json=data, verify=self.verify)
        elif self.args.method == 'DELETE':
            data = [self.params]
            return self.session.delete(self.api + 'objects', json=data, verify=self.verify)

    def objects_query(self):
        data = {
            'order': self.params.pop('order', []),
            'page': self.params.pop('page', 1),
            'pagesize': self.params.pop('pagesize', 5),
            'fields': self.params.pop('fields', []),
        }

        if self.params:
            filter = {}
            if 'from' in self.params and 'till' in self.params:
                filter['modified'] = {}
                current_time = int(time.time())

                ts_from = self.params.pop('from')
                if isinstance(ts_from, (str, bytes)) and ts_from.startswith("CURRENT_TIME"):
                    ts_from = current_time + get_time_shift_value(ts_from)
                filter['modified']['from'] = ts_from

                ts_till = self.params.pop('till')
                if isinstance(ts_till, (str, bytes)) and ts_till.startswith("CURRENT_TIME"):
                    ts_till = current_time + get_time_shift_value(ts_till)
                filter['modified']['till'] = ts_till

            filter.update(self.params)
            data['filters'] = [filter]

        return self.session.post(self.api + 'objects/query', json=data, verify=self.verify)

    def objects_sources(self):
        if self.args.method == 'POST':
            data = [self.params]
            return self.session.post(self.api + 'objects/sources', json=data, verify=self.verify)
        elif self.args.method == 'PUT':
            data = [self.params]
            return self.session.put(self.api + 'objects/sources', json=data, verify=self.verify)
        elif self.args.method == 'DELETE':
            data = [self.params]
            return self.session.delete(self.api + 'objects/sources', json=data, verify=self.verify)

    def objects_sources_query(self):
        data = {
            'order': self.params.pop('order', []),
            'page': self.params.pop('page', 1),
            'pagesize': self.params.pop('pagesize', 5),
            'fields': self.params.pop('fields', []),
        }
        if self.params:
            data['filters'] = [self.params]
        return self.session.post(self.api + 'objects/sources/query', json=data, verify=self.verify)

    def points_operate(self):
        data = {
            'sid': self.params.pop('sid', 0),
            'iess': self.params.pop('iess', ''),
            'idcs': self.params.pop('idcs', ''),
            'zd': self.params.pop('zd', ''),
        }
        if 'value' in self.params:
            data['value'] = self.params['value']
        if 'quality' in self.params:
            data['quality'] = self.params['quality']
        return self.session.post(self.api + 'points/operate', json=[data], verify=self.verify)

    def points_query(self):
        if self.args.method == 'get':
            data = {
                'source': self.params.pop('source', ''),
                'filter': self.params.pop('filter', ''),
                'order': self.params.pop('order', []),
                'page': self.params.pop('page', 1),
                'pagesize': self.params.pop('pagesize', 5),
                'fields': self.params.pop('fields', []),
            }
            return self.session.get(self.api + 'points/query', params=data, verify=False)

        data = {
            'order': self.params.pop('order', []),
            'page': self.params.pop('page', 1),
            'pagesize': self.params.pop('pagesize', 5),
            'fields': self.params.pop('fields', []),
        }
        if self.params:
            filter = {}
            current_time = int(time.time())
            if 'from' in self.params:
                ts_from = self.params.pop('from')
                if isinstance(ts_from, (str, bytes)) and ts_from.startswith("CURRENT_TIME"):
                    ts_from = current_time + get_time_shift_value(ts_from)
                filter.setdefault('ts', {})['from'] = ts_from

            if 'till' in self.params:
                ts_till = self.params.pop('till')
                if isinstance(ts_till, (str, bytes)) and ts_till.startswith("CURRENT_TIME"):
                    ts_till = current_time + get_time_shift_value(ts_till)
                filter.setdefault('ts', {})['till'] = ts_till

            filter.update(self.params)
            data['filters'] = [filter]

        return self.session.post(self.api + 'points/query', json=data, verify=self.verify)

    def points_export(self):
        if self.args.method == 'get':
            data = OrderedDict()
            data['iess'] = self.params.pop('iess','')
            data['idcs'] = self.params.pop('idcs','')
            data['desc'] = self.params.pop('desc','')
            data['aux'] = self.params.pop('aux','')
            data['ac'] = self.params.pop('ac','')
            data['rt'] = self.params.pop('rt', 'ANALOG,DOUBLE,BINARY,PACKED,INT64,TEXT')
            data['order'] = self.params.pop('order', [])
            data['separator'] = self.params.pop('separator', ' ')
            data['encoding'] = self.params.pop('encoding', 'utf-8')
            data['flags'] = self.params.pop('flags', 2)
            data['page'] = self.params.pop('page', 1)
            data['pagesize'] = self.params.pop('pagesize', 1000000)
            return self.session.get(self.api + 'points/export', params=data, verify=False)

        data = OrderedDict()
        data['order'] = self.params.pop('order', [])
        data['separator'] = self.params.pop('separator', ' ')
        data['encoding'] = self.params.pop('encoding', 'utf-8')
        data['flags'] = self.params.pop('flags', 2)
        data['page'] = self.params.pop('page', 1)
        data['pagesize'] = self.params.pop('pagesize', 1000000)
        data['fields'] = self.params.pop('fields', [])

        if self.params:
            data['filters'] = [self.params]
        return self.session.post(self.api + 'points/export', json=data, verify=self.verify)

    def points_publish(self):
        data = {
            'sid': self.params.pop('sid', 0),
            'iess': self.params.pop('iess', ''),
            'idcs': self.params.pop('idcs', ''),
            'zd': self.params.pop('zd', ''),
        }
        if 'ts' in self.params:
            data['ts'] = self.params['ts']
        if 'lts' in self.params:
            data['lts'] = self.params['lts']
        if 'value' in self.params:
            data['value'] = self.params['value']
        if 'quality' in self.params:
            data['quality'] = self.params['quality']
        if 'duration' in self.params:
            data['duration'] = self.params['duration']
        return self.session.post(self.api + 'points/publish', json=[data], verify=self.verify)

    def points_unpublish(self):
        data = {
            'sid': self.params.pop('sid', 0),
            'iess': self.params.pop('iess', ''),
            'idcs': self.params.pop('idcs', ''),
            'zd': self.params.pop('zd', ''),
        }
        return self.session.post(self.api + 'points/unpublish', json=[data], verify=self.verify)

    def points_sources(self):
        return self.session.get(self.api + 'points/sources', verify=self.verify)

    def requests(self):
        if self.args.method == 'GET':
            ids = ','.join(map(str, self.params.pop('id', [])))
            return self.session.get(self.api + 'requests?id=' + ids, verify=self.verify)
        elif self.args.method == 'DELETE':
            data = [self.params]
            return self.session.delete(self.api + 'requests', json=data, verify=self.verify)

    def events_read(self):
        fields = self.params.pop('fields', None)
        data = {
            'filter': {
                'period': {
                    'from': self.params.pop('from', int(time.time()) - 24*3600),
                    'till': self.params.pop('till', int(time.time())),
                }
            },
            'maxCount': self.params.pop('maxcount', 20),
        }
        if 'sid' in self.params:
            data['filter']['pointId'] = [{'sid': self.params.pop('sid')}]
        if 'iess' in self.params:
            data['filter']['pointId'] = [{'iess': self.params.pop('iess')}]
        data['filter'].update(self.params)
        res = self.session.post(self.api + 'events/read', json=data, verify=self.verify)
        log_request(res, pretty_print=False)
        data = res.json()
        self._wait_for_request_execution(data['id'])
        fields = '&fields=' + ','.join(fields) if fields else ''
        url = self.api + 'events/read?id=' + str(data['id']) + fields
        res = self.session.get(url, verify=self.verify)
        log_request(res)

    def trend(self):
        data = []
        items_params = {
            'period': {
                'from': self.params.get('from', int(time.time()) - 600),
                'till': self.params.get('till', int(time.time())),
            },
            'pixelCount': self.params.get('pixelCount', 20),
        }

        if 'shadePriority' in self.params:
            items_params['shadePriority'] = self.params['shadePriority']

        if isinstance(self.params['iess'], list):
            for iess in self.params['iess']:
                data.append(merge({'pointId': {'iess': iess}}, items_params))
        else:
            data.append(merge({'pointId': {'iess': self.params['iess']}}, items_params))

        res = self.session.post(self.api + 'trend', json=data, verify=self.verify)
        log_request(res, pretty_print=False)
        data = res.json()

        url = self.api + 'trend?id=' + str(data['id'])
        get_chunks = lambda: self.session.get(url, verify=self.verify)
        self._wait_for_request_execution(data['id'], get_chunks)

    def trend_tabular(self):
        items_params = {}
        if 'shadePriority' in self.params:
            items_params['shadePriority'] = self.params['shadePriority']
        if 'function' in self.params:
            items_params['function'] = self.params['function']
        if 'params' in self.params:
            items_params['params'] = self.params['params']

        data = {
            'period': {
                'from': self.params.pop('from', int(time.time()) - 600),
                'till': self.params.pop('till', int(time.time())),
            },
            'step': self.params.pop('step', 60),
            'items': []
        }

        if isinstance(self.params['iess'], list):
            for iess in self.params['iess']:
                data['items'].append(merge({'pointId': {'iess': iess}}, items_params))
        else:
            data['items'].append(merge({'pointId': {'iess': self.params['iess']}}, items_params))

        res = self.session.post(self.api + 'trend/tabular', json=data, verify=self.verify)
        log_request(res, pretty_print=False)
        data = res.json()

        url = self.api + 'trend/tabular?id=' + str(data['id'])
        get_chunks = lambda: self.session.get(url, verify=self.verify)
        self._wait_for_request_execution(data['id'], get_chunks)

    def trend_groups(self):
        return self.session.get(self.api + 'trend/groups', verify=self.verify)

    def events(self):
        data = self.params
        if 'sid' in self.params:
            data['pointId'] = {'sid': self.params.pop('sid')}
        if 'iess' in self.params:
            data['pointId'] = {'iess': self.params.pop('iess')}
        return self.session.post(self.api + 'events', json=[self.params], verify=self.verify)

    def report_configs_query(self):
        data = {
            'page': self.params.pop('page', 1),
            'pagesize': self.params.pop('pagesize', 5),
        }
        if self.params:
            objectFilter = {}
            keys = ('id', 'fileRe', 'nameRe', 'sourceNameRe', 'sourceId', 'sg', 'tg', 'md5')
            for key in keys:
                if key in self.params:
                    objectFilter[key] = self.params.pop(key)

            if 'from' in self.params and 'till' in self.params:
                objectFilter['modified'] = {}
                current_time = int(time.time())

                ts_from = self.params.pop('from')
                if isinstance(ts_from, (str, bytes)) and ts_from.startswith("CURRENT_TIME"):
                    ts_from = current_time + get_time_shift_value(ts_from)
                objectFilter['modified']['from'] = ts_from

                ts_till = self.params.pop('till')
                if isinstance(ts_till, (str, bytes)) and ts_till.startswith("CURRENT_TIME"):
                    ts_till = current_time + get_time_shift_value(ts_till)
                objectFilter['modified']['till'] = ts_till

            filter = {}
            if 'outputType' in self.params:
                filter['outputType'] = self.params.pop('outputType')

            if 'executionCondition' in self.params:
                filter['executionCondition'] = self.params.pop('executionCondition')

            if len(objectFilter.keys()) > 0:
                filter['objectFilter'] = objectFilter

            data['filters'] = [filter]
        return self.session.post(self.api + 'report/configs/query', json=data, verify=self.verify)

    def report_configs(self):
        if self.args.method == 'POST':
            data = [{
                'sourceId': self.params.pop('sourceId'),
                'rdfFileName': self.params.pop('rdfFileName'),
                'config': self.params,
            }]
            return self.session.post(self.api + 'report/configs', json=data, verify=self.verify)
        elif self.args.method == 'PUT':
            data = [{
                'id': self.params.pop('id'),
                'config': self.params,
            }]
            return self.session.put(self.api + 'report/configs', json=data, verify=self.verify)
        elif self.args.method == 'DELETE':
            data = [self.params]
            return self.session.delete(self.api + 'report/configs', json=data, verify=self.verify)

    def report_custom(self):
        data = {
            'rdf': {
                'localTime': self.params.get('localTime', True),
                'showDstTransition': self.params.get('showDstTransition', True),
                'showQuality': self.params.get('showQuality', True),
                'precision': self.params.get('precision', 6),
                'timeMode': self.params.get('timeMode', 'RELATIVE'),
                'addressingType': self.params.get('addressingType', 'A1'),
                'shadePriority': self.params.get('shadePriority', 'DEFAULT'),
                'rows': self.params['rows']
            },
            'dtRef': self.params.get('dtRef', int(time.time()))
        }
        res = self.session.post(self.api + 'report/custom', json=data, verify=self.verify)
        log_request(res)
        data = res.json()
        self._wait_for_request_execution(data['id'])
        url = self.api + 'report/custom?id=' + str(data['id'])
        res = self.session.get(url, verify=self.verify)
        log_request(res)

    def report_global(self):
        data = {
            'sourceId': self.params['sourceId'],
            'file': self.params['file'],
            'rdf': {
                'localTime': self.params.get('localTime', True),
                'showDstTransition': self.params.get('showDstTransition', True),
                'showQuality': self.params.get('showQuality', True),
                'precision': self.params.get('precision', 6),
                'timeMode': self.params.get('timeMode', 'RELATIVE'),
                'addressingType': self.params.get('addressingType', 'A1'),
                'shadePriority': self.params.get('shadePriority', 'DEFAULT'),
                'rows': self.params['rows']
            },
        }
        if 'name' in self.params:
            data['name'] = self.params['name']
        res = self.session.post(self.api + 'report/global', json=data, verify=self.verify)
        log_request(res)

    def report_global_run(self):
        data = {
            'configId': self.params['configId'],
            'dtRef': self.params.get('dtRef', int(time.time()))
        }
        res = self.session.post(self.api + 'report/global/run', json=data, verify=self.verify)
        log_request(res, pretty_print=False)
        data = res.json()
        self._wait_for_request_execution(data['id'])

    def shades_points(self):
        data = {
            'order': self.params.pop('order', []),
            'page': self.params.pop('page', 1),
            'pagesize': self.params.pop('pagesize', 10),
        }
        if self.params:
            data['filters'] = [self.params]
        return self.session.post(self.api + 'shades/points', json=data, verify=self.verify)

    def shades_read(self):
        data = []
        items_params = {
            'period': {
                'from': self.params.get('from', int(time.time()) - 600),
                'till': self.params.get('till', int(time.time())),
            },
        }

        if isinstance(self.params['iess'], list):
            for iess in self.params['iess']:
                data.append(merge({'pointId': {'iess': iess}}, items_params))
        else:
            data.append(merge({'pointId': {'iess': self.params['iess']}}, items_params))

        res = self.session.post(self.api + 'shades/read', json=data, verify=self.verify)
        log_request(res, pretty_print=False)
        data = res.json()

        self._wait_for_request_execution(data['id'])

        url = self.api + 'shades/read?id=' + str(data['id'])
        res = self.session.get(url, verify=self.verify)
        log_request(res)

    def shades_write(self):
        data = {
            'period': {
                'from': self.params.get('from', int(time.time()) - 600),
                'till': self.params.get('till', int(time.time())),
            },
            'value': self.params.get('value'),
            'quality': self.params.get('quality'),
        }
        if 'sid' in self.params:
            data['pointId'] = {'sid': self.params.pop('sid')}
        if 'iess' in self.params:
            data['pointId'] = {'iess': self.params.pop('iess')}

        res = self.session.post(self.api + 'shades/write', json=[data], verify=self.verify)
        log_request(res, pretty_print=False)
        data = res.json()
        self._wait_for_request_execution(data['id'])

    def shades_clear(self):
        data = {
            'period': {
                'from': self.params.get('from', int(time.time()) - 600),
                'till': self.params.get('till', int(time.time())),
            }
        }
        if 'sid' in self.params:
            data['pointId'] = {'sid': self.params.pop('sid')}
        if 'iess' in self.params:
            data['pointId'] = {'iess': self.params.pop('iess')}

        res = self.session.post(self.api + 'shades/clear', json=[data], verify=self.verify)
        log_request(res, pretty_print=False)
        data = res.json()
        self._wait_for_request_execution(data['id'])

    def shades_copy(self):
        data = {
            'period': {
                'from': self.params.get('from', int(time.time()) - 600),
                'till': self.params.get('till', int(time.time())),
            }
        }
        if 'srcSid' in self.params:
            data['srcPointId'] = {'sid': self.params.pop('srcSid')}
        if 'srcIess' in self.params:
            data['srcPointId'] = {'iess': self.params.pop('srcIess')}

        if 'dstSid' in self.params:
            data['dstPointId'] = {'sid': self.params.pop('dstSid')}
        if 'dstIess' in self.params:
            data['dstPointId'] = {'iess': self.params.pop('dstIess')}

        res = self.session.post(self.api + 'shades/copy', json=[data], verify=self.verify)
        log_request(res, pretty_print=False)
        data = res.json()
        self._wait_for_request_execution(data['id'])

    def tg(self):
        return self.session.get(self.api + 'tg', verify=self.verify)

    def sg(self):
        return self.session.get(self.api + 'sg', verify=self.verify)

    def user_sg(self):
        return self.session.get(self.api + 'user/sg', verify=self.verify)

    def user_profile(self):
        return self.session.get(self.api + 'user/profile', verify=self.verify)

    def users_query(self):
        data = {
            'order': self.params.pop('order', []),
            'page': self.params.pop('page', 1),
            'pagesize': self.params.pop('pagesize', 10),
            'fields': self.params.pop('fields', []),
        }
        if self.params:
            data['filters'] = [self.params]
        return self.session.post(self.api + 'users/query', json=data, verify=self.verify)

    def status(self):
        return self.session.get(self.api + 'status', verify=self.verify)

    def license(self):
        return self.session.get(self.api + 'license', verify=self.verify)

    def diagram_open(self):
        return self.session.post(self.api + 'diagram/open', json=self.params, verify=self.verify)

    def diagram_close(self):
        uuid = self.params.pop('uuid', None)
        if uuid:
            return self.session.post(self.api + uuid + "/close", json=self.params, verify=self.verify)
        raise Exception("UUID is required to close diagram")

    def diagram_resize(self):
        uuid = self.params.pop('uuid', None)
        if uuid:
            return self.session.post(self.api + uuid + "/resize", json=self.params, verify=self.verify)
        raise Exception("UUID is required to resize diagram")

    def diagram_viewport(self):
        uuid = self.params.pop('uuid', None)
        if uuid:
            return self.session.post(self.api + uuid + "/viewport", json=self.params, verify=self.verify)
        raise Exception("UUID is required to set diagram viewport")

    def _wait_for_request_execution(self, req_id, get_chunks=None):
        progress = 0
        st = time.time()
        while True:
            time.sleep(0.2)
            res = self.session.get(self.api + 'requests?id=' + str(req_id), verify=self.verify)
            if not self._noRequestExecutionLogs:
                log_request(res, verbose=False, cookies=False, pretty_print=False)
            status = res.json()[str(req_id)]
            if status['status'] == 'FAILURE':
                raise RuntimeError('Request failed: ' + status['message'])
            elif status['status'] == 'SUCCESS':
                if get_chunks:
                    res = get_chunks()
                    log_request(res, floatPrecision=self._floatPrecision)
                break
            elif status['status'] == 'EXECUTING':
                if get_chunks and status['progress'] != progress:
                    res = get_chunks()
                    if not self._noRequestExecutionLogs:
                        log_request(res)
        print('req_id = {}, executed in: {:.3f} s\n'.format(req_id, time.time() - st))

    def _is_expected_result(self, res):
        expectedMatchCount = self.args.expectedMatchCount
        if expectedMatchCount is None:
            return True

        if res is None or not res.text:
            return True

        json_data = {}
        try:
            json_data = res.json()
        except requests.JSONDecodeError:
            print("Couldn’t decode the text into json")

        if 'matchCount' not in json_data.keys():
            return True

        return expectedMatchCount == json_data['matchCount']


def merge(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))  # python 2 compatible syntax

def getSortedStringFromDictionary(dictionary):
     return ('{' + ', '.join("'" + k + "': '" + v + "'" for k, v in
         sorted(dictionary.items(), key=lambda x: x[0], reverse=False)) + '}')

def print_json_pretty(inputJson, floatPrecisionCall=None):
    return json.dumps(json.loads(inputJson, object_pairs_hook=OrderedDict, parse_float=floatPrecisionCall),
                      ensure_ascii=False,
                      indent=4,
                      separators=(',', ': '),
                      sort_keys=True)

def log_request(res, verbose=True, cookies=True, pretty_print=True, floatPrecision=None):
    floatPrecisionCall = None
    if floatPrecision:
        floatPrecisionCall = lambda numf: round(float(numf), floatPrecision)

    print('<<< Request: {} {}'.format(res.request.method, res.url))
    if verbose:
        headers = res.request.headers
        headers.pop('User-Agent')
        print('    [headers]: ' + getSortedStringFromDictionary(headers))
    if res.request.body:
        if pretty_print:
            print('    [body]: <pretty-print>\n' + print_json_pretty(res.request.body, floatPrecisionCall))
        else:
            print('    [body]: ' + str(res.request.body.decode('utf-8')))
    print('>>> Response: ' + str(res.status_code))
    if verbose:
        print('    [headers]: ' + getSortedStringFromDictionary(headers))
    if cookies and res.cookies:
        print('    [cookies]: ' + str(res.cookies.get_dict()))
    if res.headers.get('content-type', '').startswith('application/json') and pretty_print:
        print('    [body]: <pretty-print>\n' + print_json_pretty(res.content, floatPrecisionCall))
    elif res.content:
        print('    [body]: ' + res.content.decode('utf-8'))
    print('')


def read_extra_params(extra_args):
    params = {}
    for extra in extra_args:
        if '=' not in extra:
            print('Invalid argument: %s' % extra)
            exit(-1)
        name, val = extra.split('=', 1)
        if name in params:
            if type(params[name]) == list:
                params[name].append(read_value(val))
            else:
                params[name] = [params[name], read_value(val)]
        elif val.startswith('[') and val.endswith(']'):
            params[name] = [read_value(v) for v in val[1:-1].split(',')]
            v = [read_value(v) for v in val[1:-1].split(',')]
        elif val.startswith("json:"):
            json_data = json.loads(val[5:])
            params[name] = json_data
        else:
            params[name] = read_value(val)
    return params


def read_value(val):
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    if val=="false" or val=="False":
        return False
    if val=="true" or val=="True":
        return True
    if val.isdigit():
        return int(val)
    try:
        float_val = float(val)
        return float_val if math.isfinite(float_val) else val
    except ValueError:
        return val


def get_time_shift_value(time_str):
    if time_str == "CURRENT_TIME":
        return 0

    m = re.match("CURRENT_TIME([-+][0-9]+)", time_str)
    if m:
        return int(m.group(1))

    print ("Invalid CURRENT_TIME format")


def is_valid_response(response):
    if response.status_code == 500:
        return False

    jsonData = {}
    if response.text:
        try:
            jsonData = json.loads(response.text)
        except json.decoder.JSONDecodeError:
            pass

    errorCode = 0
    errorMessage = None
    if isinstance(jsonData, dict):
        errorCode = jsonData.get('error', 0)
        errorMessage = jsonData.get('message', None)

    if (errorCode == 30 and errorMessage == "Points\' data not synchronized yet.") or \
       (errorCode == 31 and errorMessage == "Point not found"):
        return False
    return True


if __name__ == '__main__':
    args, extra_args = parser.parse_known_args()
    params = read_extra_params(extra_args)
    client = Client(args, params)

    if args.sessionid or args.endpoint.startswith('/') or args.endpoint in ('test', 'login', 'authenticate'):
        client.run(args.endpoint)
    else:
        client.run('login', verbose=False)
        client.run(args.endpoint)
        client.run('logout', verbose=False)
