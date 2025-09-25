#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Niniejsze dane stanowią tajemnicę przedsiębiorstwa.
# The contents of this file is proprietary information.

import sys
import os
import time
import calendar
import http.client
import urllib.request, urllib.error, urllib.parse
import urllib.parse
import base64
import suds
import suds.client
import json

from datetime import datetime

config = {"wsdlUrl": ["http://localhost:43080/eds.wsdl"],
          "httpUrl": ["http://localhost:43080"]}
client = None


class ParamRequiredError(RuntimeError):
    pass


def param(name, *args, **kwargs):
    if name not in config:
        if len(args):
            return args[0]
        if 'default' in kwargs:
            return kwargs['default']
        raise ParamRequiredError(name)
    return config[name][0]


def create_time_period(prefix="", no=0):
    period = client.factory.create("TimePeriod")
    if prefix + "from" in config and len(config[prefix + "from"]) > no and \
            prefix + "till" in config and len(config[prefix + "till"]) > no:
        period.__dict__["from"] = create_timestamp(config[prefix + "from"][no])
        period.__dict__["till"] = create_timestamp(config[prefix + "till"][no])
    else:
        period.__dict__["from"].second = int(time.time()) - 3600
        period.__dict__["till"].second = int(time.time())
    return period


def create_timestamp(date_str):
    ts = client.factory.create("Timestamp")
    ts.second = calendar.timegm(time.strptime(date_str, "%Y-%m-%dT%H:%M:%S"))
    return ts


def create_struct(name, prefix="", no=0):
    struct = client.factory.create(name)
    for key in struct.__dict__:
        if prefix + key in config:
            struct.__dict__[key] = config[prefix + key][no]
    return struct


def wait_for_request_execution(req_id):
    st = time.time()
    while True:
        status = client.service.getRequestStatus(param('authString'), req_id)
        if status.status == "REQUEST-FAILURE":
            raise RuntimeError('Request failed: ' + str(status))
        elif status.status == "REQUEST-SUCCESS":
            break
        elif status.status == "REQUEST-EXECUTING":
            sys.stderr.write("%3.0f%%\n" % float(status.progress))
        time.sleep(0.5)
    #print('Request executed in: %d s' % int(time.time() - st + 0.5))


def getServerTime():
    return client.service.getServerTime(param('authString'))


def login():
    return client.service.login(param('user'), param('password'))


def ping():
    return client.service.ping(param('authString'))


def logout():
    return client.service.logout(param('authString'))


def getSecurityGroups():
    return client.service.getSecurityGroups(param('authString'))


def getUserSecurityGroups():
    return client.service.getUserSecurityGroups(param('authString'))


def getTechnologicalGroups():
    return client.service.getTechnologicalGroups(param('authString'))


def getServerTime():
    return client.service.getServerTime(param('authString'))


def getServerStatus():
    return client.service.getServerStatus(param('authString'))


def getLicenseInfo():
    return client.service.getLicenseInfo(param('authString'))


def getPoints():
    return client.service.getPoints(param('authString'),
                                    create_struct("PointFilter"),
                                    param('order', None),
                                    param('startIdx', None),
                                    param('maxCount', None))


def getPointsWithCustomFilter():
    return client.service.getPointsWithCustomFilter(param('authString'),
                                                    param('source'),
                                                    param("pointFilterName"),
                                                    param('order', None),
                                                    param('startIdx', None),
                                                    param('maxCount', None))


def getPointsSources():
    return client.service.getPointsSources(param('authString'))


def getModifiedPoints():
    timestamp = create_struct("Timestamp")
    return client.service.getModifiedPoints(param('authString'),
                                            param('startIdx', None),
                                            param('maxCount', None),
                                            timestamp,
                                            param('dbSequence', None))


def operatePoints():
    point = client.factory.create("PointNewValue")
    point.id = create_struct("PointId")
    point.value = create_struct("PointValue")
    point.quality = param('quality', None)
    return client.service.operatePoints(param('authString'), [point])


def publishPoints():
    point = client.factory.create("PointNewValue")
    point.id = create_struct("PointId")
    point.value = create_struct("PointValue")
    point.quality = param('quality', None)
    point.duration.seconds = param('duration', None)
    points = client.factory.create("PointsNewValues")
    points.point = [point]
    return client.service.publishPoints(param('authString'), points)


def unpublishPoints():
    pointid = create_struct("PointId")
    return client.service.unpublishPoints(param('authString'), pointid)


def getTrend():
    request = create_struct("TrendRequest")
    request.period = create_time_period()
    request.pixelCount = param('pixelCount', 100)
    requestItem = create_struct("TrendRequestItem")
    requestItem.pointId = create_struct("PointId")
    del requestItem.shadePriority
    request.items.append(requestItem)
    req_id = client.service.requestTrend(param('authString'), request)
    wait_for_request_execution(req_id)
    return client.service.getTrend(param('authString'), req_id)


def getTrendGroups():
    response = client.service.getTrendGroups(param('authString'),
                                             param('configurationVersion', None))
    file_name = 'trend_groups-{}.json'.format(int(time.time() + 0.5))

    with open(file_name, "wb") as f:
        f.write(response)

    sys.stderr.write(
        "Trend groups saved to `{}` file ({:n}B).\n".format(file_name, len(response)))


def getTabular():
    request = client.factory.create("TabularRequest")
    request.period = create_time_period()
    request.step.seconds = param('step', 60)
    requestItem = create_struct("TabularRequestItem")
    requestItem.pointId = create_struct("PointId")
    del requestItem.shadePriority
    if "param" in config:
        for val in config["param"].values():
            requestItem.param.append(val)

    request.items.append(requestItem)
    req_id = client.service.requestTabular(param('authString'), request)
    wait_for_request_execution(req_id)
    return client.service.getTabular(param('authString'), req_id)


def getTabularCSV():
    request = client.factory.create("TabularRequest")
    request.period = create_time_period()
    request.step.seconds = param('step', 60)
    req_function = config['function']

    points = config['point']

    for point_name in points:
        requestItem = create_struct("TabularRequestItem")
        requestItem.pointId = create_struct("PointId")
        requestItem.pointId.iess = point_name
        del requestItem.shadePriority
        if "param" in config:
            for val in config["param"].values():
                requestItem.param.append(val)

        requestItem.function = req_function
        request.items.append(requestItem)

    req_id = client.service.requestTabular(param('authString'), request)
    wait_for_request_execution(req_id)
    result = client.service.getTabular(param('authString'), req_id)
    header_items = ["time"]
    for pt_data in result.pointsIds:
        #print(f"{pt_data.iess}")
        header_items.append(f"{pt_data.iess}:{req_function}")
    print(",".join(header_items))
    for row in result.rows:
        row_items = [f"{row.ts.second}"]
        for value in row.values:
             row_items.append(f"{value.value.dav}")
        print(",".join(row_items))
    return
    #return result


def getShade():
    request = client.factory.create("ShadeSelector")
    request.pointId = create_struct("PointId")
    request.period = create_time_period()
    req_id = client.service.requestShadesRead(param('authString'), [request])
    wait_for_request_execution(req_id)
    return client.service.getShades(param('authString'), req_id)


def writeShade():
    shades = []
    shade = client.factory.create("Shade")
    shade.pointId = create_struct("PointId")
    shadeValue = create_struct("ShadeValue")
    shadeValue.period = create_time_period()
    shadeValue.value = create_struct("PointValue")
    shade.values.append(shadeValue)
    shades.append(shade)
    req_id = client.service.requestShadesWrite(param('authString'), shades)
    wait_for_request_execution(req_id)


def clearShade():
    request = client.factory.create("ShadesClearRequest")
    item = client.factory.create("ShadesClearRequestItem")
    item.pointId = create_struct("PointId")
    item.period = create_time_period()
    request.item.append(item)
    req_id = client.service.requestShadesClear(param('authString'), request)
    wait_for_request_execution(req_id)


def copyShade():
    request = client.factory.create("ShadesCopyRequest")
    item = client.factory.create("ShadesCopyRequestItem")
    item.srcPointId = create_struct("PointId", "src.")
    item.dstPointId = create_struct("PointId", "dst.")
    item.period = create_time_period()
    request.item.append(item)
    req_id = client.service.requestShadesCopy(param('authString'), request)
    wait_for_request_execution(req_id)


def getEvents():
    event_filter = create_struct("EventFilter")
    event_filter.period = create_time_period()
    req_id = client.service.requestEvents(param('authString'), event_filter)
    wait_for_request_execution(req_id)
    return client.service.getEvents(param('authString'), req_id)


def getReportsConfigs():
    conf_filter = create_struct('ReportConfigFilter')
    conf_filter.objectFilter = create_struct('ObjectFilter')
    return client.service.getReportsConfigs(param('authString'), conf_filter)


def createReportConfig():
    path = os.path.dirname(os.path.realpath(__file__))
    repConfig = create_struct('ReportConfigDefinition')
    repConfig.runDelay.seconds = 10
    repConfig.timeMaskExpression = '0 * * * *'
    repConfig.inputValues = '0;'
    repConfig.outputMaskFileTxt = param(
        'outputMaskFileTxt', os.path.join(
            path, 'rep_test.txt'))
    repConfig.outputMaskFileRdf = param(
        'outputMaskFileRdf', os.path.join(
            path, 'rep_test.rdf'))
    repConfig.outputMaskFileEdf = param(
        'outputMaskFileEdf', os.path.join(
            path, 'rep_test.edf'))
    repConfig.outputMaskFileHtml = param(
        'outputMaskFileHtml', os.path.join(
            path, 'rep_test.html'))
    repConfig.outputMaskFilePdf = param(
        'outputMaskFilePdf', os.path.join(
            path, 'rep_test.pdf'))
    repConfig.outputMaskFileCsv = param(
        'outputMaskFileCsv', os.path.join(
            path, 'rep_test.csv'))
    repConfig.outputMaskDatabaseRdf = param(
        'outputMaskDatabaseRdf', 'rep_test.rdf')
    repConfig.outputMaskDatabaseEdf = param(
        'outputMaskDatabaseEdf', 'rep_test.edf')
    repConfig.outputMaskDatabaseHtml = param(
        'outputMaskDatabaseHtml', 'rep_test.html')
    return client.service.createReportConfig(param('authString'),
                                             sourceName=param(
                                                 'sourceName', 'test_local'),
                                             rdfFileName=param(
        'rdfFileName', 'test_webapi_client.rdf'),
        config=repConfig)


def alterReportConfig():
    path = os.path.dirname(os.path.realpath(__file__))
    repConfig = create_struct('ReportConfigDefinition')
    repConfig.runDelay.seconds = 10
    repConfig.outputMaskFileTxt = param(
        'outputMaskFileTxt', os.path.join(
            path, 'rep_test_altered.txt'))
    repConfig.outputMaskFileRdf = param(
        'outputMaskFileRdf', os.path.join(
            path, 'rep_test_altered.rdf'))
    repConfig.outputMaskFileEdf = param(
        'outputMaskFileEdf', os.path.join(
            path, 'rep_test_altered.edf'))
    repConfig.outputMaskFileHtml = param(
        'outputMaskFileHtml', os.path.join(
            path, 'rep_test_altered.html'))
    repConfig.outputMaskFilePdf = param(
        'outputMaskFilePdf', os.path.join(
            path, 'rep_test_altered.pdf'))
    repConfig.outputMaskFileCsv = param(
        'outputMaskFileCsv', os.path.join(
            path, 'rep_test_altered.csv'))
    repConfig.outputMaskDatabaseRdf = param(
        'outputMaskDatabaseRdf', 'rep_test_altered.rdf')
    repConfig.outputMaskDatabaseEdf = param(
        'outputMaskDatabaseEdf', 'rep_test_altered.edf')
    repConfig.outputMaskDatabaseHtml = param(
        'outputMaskDatabaseHtml', 'rep_test_altered.html')
    return client.service.alterReportConfig(param('authString'),
                                            configId=param('configId'),
                                            config=repConfig)


def deleteReportConfig():
    return client.service.deleteReportConfig(
        param('authString'), param('configId'))


def requestGlobalReport():
    request = create_struct("GlobalReportRequest")
    request.dtRef.second = int(param('dtref', int(time.time())))
    req_id = client.service.requestGlobalReport(param('authString'), request)
    wait_for_request_execution(req_id)


def requestCustomReport():
    request = create_struct('CustomReportRequest')
    rdf = create_struct('ReportDefinition')
    rdf.timeMode = param('timeMode', None)
    if rdf.timeMode is None:
        del rdf.timeMode
    del rdf.addressingType
    del rdf.shadesPriority
    row = create_struct('ReportDefinitionRow')
    row.cells = [
        cell(
            param(
                'content',
                '=1.2G')),
        cell('=@DT_REF()-3600'),
        cell('=@DT_REF()'),
        cell('=@DT_REF()+3600')]
    row2 = create_struct('ReportDefinitionRow')
    row2.cells = [cell('zażółć'), cell('gęślą'), cell('jaźń')]
    row3 = create_struct('ReportDefinitionRow')
    if hasattr(rdf, 'timeMode') and rdf.timeMode == 'REPORT-TIME-MODE-ABSOLUTE':
        row3.cells = [
            cell('=@VALUE(\\A1\\, @DT_REF()-3600)'),
            cell('=@VALUE(\\A2\\, @DT_REF()-7200)')]
    else:
        row3.cells = [
            cell('=@VALUE(\\A1\\, -3600)'),
            cell('=@VALUE(\\A2\\, -7200)')]
    rdf.rows = [row, row2, row3]
    request.rdf = rdf
    request.dtRef.second = int(param('dtref', int(time.time())))
    req_id = client.service.requestCustomReport(param('authString'), request)
    wait_for_request_execution(req_id)
    return client.service.getCustomReport(param('authString'), req_id)


def createGlobalReport():
    rdf = create_struct('ReportDefinition')
    del rdf.timeMode
    del rdf.addressingType
    del rdf.shadesPriority
    row = create_struct('ReportDefinitionRow')
    row.cells = [
        cell('=1.2G'),
        cell('=@DT_REF()-3600'),
        cell('=@DT_REF()'),
        cell('=@DT_REF()+3600')]
    row2 = create_struct('ReportDefinitionRow')
    row2.cells = [cell('zażółć'), cell('gęślą'), cell('jaźń')]
    rdf.rows = [row, row2]
    return client.service.createGlobalReport(param('authString'),
                                             None,
                                             param('sourceName', 'test_local'),
                                             param(
        'file', 'test_webapi_client.rdf'),
        param('name', None),
        rdf,
        [0, 1])


def cell(content):
    cell = create_struct('ReportDefinitionCell')
    cell.content = content
    return cell


def getObjectsSources():
    return client.service.getObjectsSources(param('authString'),
                                            create_struct(
                                                "ObjectSourceFilter"),
                                            param('order', None),
                                            param('startIdx', None),
                                            param('maxCount', None))


def getObject():
    url = '%s/objects/%s/%s?authString=%s' \
          % (param('httpUrl'), param('source'), param('file'), param('authString'))

    print('Downloadnig %s/%s to %s' % (param('source'), param('file'), param('file')))
    print('URL: ' + url)

    response = urllib.request.urlopen(url.encode('utf-8'))
    with open(param('save', param('file')), 'wb') as f:
        f.write(response.read())

    if response.getcode() != 200:
        print('RESPONSE CODE: %s' % response.getcode())
        print('\nRESPONSE HEADERS:\n%s' % response.info())
        print('\nRESPONSE DATA: %s' % response.read().strip())


def putObject():
    print('Uploading "%s" file  to "%s/%s"' % (param('file'), param('source'), param('file')))

    url = '%s/objects/%s/%s?authString=%s' \
          % (param('httpUrl'), param('source'), param('file'), param('authString'))

    if 'sg' in config:
        url += '&sg=' + param('sg')

    if 'tg' in config:
        url += '&tg=' + param('tg')

    if 'objectName' in config:
        url += '&objectName=' + param('objectName')

    print('URL: ' + url)

    data = open(param('file'), 'rb').read()
    length = os.path.getsize(param('file'))

    # url and data must be of the same type so url must be str not unicode
    request = urllib.request.Request(url.encode('utf-8'), data=data)
    request.add_header('Content-Type', 'application/octet-stream')
    request.add_header('Content-Length', '%d' % length)
    request.get_method = lambda: 'PUT'

    response = urllib.request.urlopen(request)

    if response.getcode() != 201:
        print('RESPONSE CODE: %s' % response.getcode())
        print('\nRESPONSE HEADERS:\n%s' % response.info())
        print('\nRESPONSE DATA: %s' % response.read().strip())


def delObject():
    url = '%s/objects/%s/%s?authString=%s' \
          % (param('httpUrl'), param('source'), param('file'), param('authString'))

    print('Deleting %s/%s' % (param('source'), param('file')))
    print('URL:\n%s' % url)

    request = urllib.request.Request(url.encode('utf-8'))
    request.get_method = lambda: 'DELETE'
    opener = urllib.request.build_opener(urllib.request.HTTPHandler(debuglevel=1))
    response = opener.open(request)

    if response.getcode() != 204:
        print('RESPONSE CODE: %s' % response.getcode())
        print('\nRESPONSE HEADERS:\n%s' % response.info())
        print('\nRESPONSE DATA: %s' % response.read().strip())


def getObjectsMetadata():
    return client.service.getObjectsMetadata(param('authString'),
                                             create_struct("ObjectFilter"),
                                             param('order', None),
                                             param('startIdx', None),
                                             param('maxCount', 100))


def getObjectsSources():
    return client.service.getObjectsSources(param('authString'),
                                            create_struct(
                                                "ObjectSourceFilter"),
                                            param('order', None),
                                            param('startIdx', None),
                                            param('maxCount', None))


def alterObject():
    attrs = client.factory.create("AlterableObjectAttributes")
    attrs.name = param('newName', None)
    attrs.sg = param('newSG', None)
    attrs.tg = param('newTG', None)
    return client.service.alterObject(
        param('authString'), 0, param('sourceName'), param('file'), attrs)


def alterObjectSource():
    attrs = client.factory.create("AlterableObjectSourceAttributes")
    attrs.desc = param('newDesc', None)
    attrs.sg = param('newSG', None)
    attrs.tg = param('newTG', None)
    return client.service.alterObjectSource(
        param('authString'), 0, param('sourceName'), attrs)


def runScript():

    def print_outputs(outputs):
        if outputs.output:
            sys.stdout.write(base64.b64decode(outputs.output))
            sys.stdout.flush()
        if outputs.errors:
            sys.stderr.write(base64.b64decode(outputs.errors))
            sys.stderr.flush()

    req_id = client.service.requestScriptRun(
        param('authString'), param('scriptName'))
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        print_outputs(
            client.service.interactWithScript(
                param('authString'),
                req_id,
                base64.b64encode(line)))

    print_outputs(
        client.service.interactWithScript(
            param('authString'),
            req_id,
            base64.b64encode(line),
            True))
    return client.service.getRequestStatus(param('authString'), req_id)


def getDiagram():
    refr = create_struct("TimeDuration")
    refr.seconds = param('refreshRate', default=3)

    url = client.service.openDiagram(
        param('authString'),
        param('source'),
        param('file'),
        refr,
        param('previousUrl', default=''),
        param('pointGroup', default=''),
        param('httpUrl', default=''))
    sys.stderr.write(url + '\n')
    if "width" in config and "height" in config:
        resolution = create_struct("DiagramResolution")
        client.service.resizeDiagram(
            param('authString'),
            url,
            "DIAGRAM-ROLE-MAIN",
            resolution)

    features = client.service.getGfxSrvFeatures(param('authString'), url)
    sys.stderr.write("GfxSrv features: " + str(features) + "\n")

    viewport = client.factory.create("DiagramArea")
    viewport.topLeft.x = 0
    viewport.topLeft.y = 0
    viewport.bottomRight.x = 1
    viewport.bottomRight.y = 1

    orgUrl = url
    url = urllib.parse.urlparse(url)

    prettyPrintMetadata = param('prettyMeta', None)

    client.service.setDiagramViewport(
        param('authString'),
        orgUrl,
        "DIAGRAM-ROLE-MAIN",
        viewport)

    time.sleep(0.2)

    tag = '0'

    print((str(datetime.now()) + ': First diagram request!'))

    try:
        while True:
            connection = http.client.HTTPConnection(url.hostname, url.port)
            connection.request("GET", url.path + "tag?_=" +
                               str(int(time.time() + 0.5)))
            current_tag = connection.getresponse().read()

            if current_tag == tag:
                continue
            tag = current_tag

            print((str(datetime.now()) + ': new diagram tag = ' + tag))

            connection.request("GET", url.path + "meta.json?tag=" + tag)
            metas = connection.getresponse().read()
            meta = json.loads(metas)

            file_name = '{}{}-meta.json'.format(param('file'),
                                                int(time.time() + 0.5))
            with open(file_name, "wb") as f:
                if prettyPrintMetadata:
                    json.dump(meta, f, indent=2, separators=(',', ': '))
                else:
                    f.write(metas)
            sys.stderr.write(
                "Diagram meta data saved to `" +
                file_name +
                "` file.\n")

            available_diagrams = []

            def getRoleIdFileName(role, data, enforced_id=None):
                return (role,
                        enforced_id if enforced_id is not None else data.get(
                            'id', ""),
                        data.get('fileName', None))

            if 'main' in meta and meta['main']:
                available_diagrams.append(
                    getRoleIdFileName(
                        'main', meta['main']))
            if 'subwindow' in meta and meta['subwindow']:
                available_diagrams.append(
                    getRoleIdFileName(
                        'subwindow', meta['subwindow']))
            if 'window' in meta and meta['window']:
                available_diagrams.append(
                    getRoleIdFileName(
                        'window',
                        meta['window'],
                        enforced_id=""))
                available_diagrams.append(
                    getRoleIdFileName(
                        'window', meta['window']))

            for extraWindow in meta.get('extraWindows', []):
                available_diagrams.append(
                    getRoleIdFileName(
                        'window', extraWindow))

            for role_id_diagram in available_diagrams:
                request_url = '{}{}{}.png?tag={}'.format(
                    url.path, role_id_diagram[0], role_id_diagram[1], tag)
                connection.request("GET", request_url)
                response = connection.getresponse()

                if response.status != http.client.OK:
                    if response.status == http.client.SEE_OTHER:
                        sys.stderr.write("Tag {} is outdated, gfx_srv no longer has {}{}.png for this tag.\n"
                                         .format(tag, role_id_diagram[0], role_id_diagram[1]))
                    sys.stderr.write("Got response '{} {}' for {}{}.png. Skipping other images.\n"
                                     .format(response.status, response.reason, role_id_diagram[0], role_id_diagram[1]))
                    break

                file_name = '{}{}-{}-{}{}.png'.format(param('file'),
                                                      int(time.time() + 0.5),
                                                      role_id_diagram[2],
                                                      role_id_diagram[0],
                                                      role_id_diagram[1])

                body = response.read()
                with open(file_name, "wb") as f:
                    f.write(body)
                sys.stderr.write(
                    "Diagram view saved to `{}` file ({:n}B).\n".format(
                        file_name, len(body)))

        time.sleep(1)

    except KeyboardInterrupt:
        client.service.closeDiagram(param('authString'), orgUrl)


def setDiagramEntryFieldValue():
    optionalWindowId = param('windowId', None)
    if optionalWindowId:
        client.service.setDiagramEntryFieldValue(param('authString'),
                                                 param('url'),
                                                 param('role'),
                                                 param('areaId'),
                                                 param('value'),
                                                 optionalWindowId)
    else:
        client.service.setDiagramEntryFieldValue(param('authString'),
                                                 param('url'),
                                                 param('role'),
                                                 param('areaId'),
                                                 param('value'))


def clickDiagram():
    optionalWindowId = param('windowId', None)
    if optionalWindowId:
        client.service.handleDiagramClick(param('authString'),
                                          param('url'),
                                          param('role'),
                                          param('areaId'),
                                          optionalWindowId)
    else:
        client.service.handleDiagramClick(param('authString'),
                                          param('url'),
                                          param('role'),
                                          param('areaId'))


def lockWindowDiagram():
    client.service.lockWindowDiagram(param('authString'),
                                     param('url'),
                                     param('windowId'),
                                     param('locked'))


def setActiveWindowDiagram():
    client.service.setActiveWindowDiagram(param('authString'),
                                          param('url'),
                                          param('windowId'))


def closeDiagram():
    optionalRole = param('role', None)
    optionalWindowId = param('windowId', None)

    if optionalRole:
        if optionalWindowId:
            client.service.closeDiagram(param('authString'),
                                        param('url'),
                                        optionalRole,
                                        optionalWindowId)
        else:
            client.service.closeDiagram(param('authString'),
                                        param('url'),
                                        optionalRole)
    else:
        client.service.closeDiagram(param('authString'),
                                    param('url'))


def dropRequest():
    return client.service.dropRequest(param('authString'), param('requestId'))


def getServerConfig():
    return client.service.getServerConfig(param('authString'))


def getLicense():
    return client.service.getLicense(param('authString'))


def openWindowDiagram():
    optionalPointGroup = param('pointGroup', None)
    if optionalPointGroup:
        client.service.openWindowDiagram(param('authString'),
                                         param('mainUrl'),
                                         param('sourceName'),
                                         param('fileName'),
                                         param('pointGroup'))
    else:
        client.service.openWindowDiagram(param('authString'),
                                         param('mainUrl'),
                                         param('sourceName'),
                                         param('fileName'))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("WebApi SOAP Client.\n")
        sys.stderr.write(
            "Usage: '%s' FUNCTION [ARG=VAL [ARG=VAL ...]]\n" %
            sys.argv[0])
        sys.exit(2)

    function = sys.argv[1]
    for arg in sys.argv[2:]:
        argval = arg.split("=", 2)
        if len(argval) != 2:
            sys.stderr.write("Invalid argument `%s`." % arg)
            sys.exit(2)
        if argval[0] not in config:
            config[argval[0]] = []
        config[argval[0]].append(argval[1])

    sys.stderr.write("GET " + config["wsdlUrl"][-1] + "\n")
    client = suds.client.Client(config["wsdlUrl"][-1])

    if 'authString' not in config:
        sys.stderr.write('login()\n')
        config['user'] = config.get('user', ['admin'])
        config['password'] = config.get('password', [''])
        config['authString'] = [login()]
        sys.stderr.write('authString: ' + config['authString'][0] + '\n')

    sys.stderr.write(function + '()\n')
    try:
        print(eval(function + '()'))
    except ParamRequiredError as e:
        print('Error: missing required param:', e)
