# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:31:11 2022

@author: ljeantet
"""


#Enregistre en fonction du nom du fichier la plage horaire à concerver pour prédire et la date et heure du fichier
import datetime

def Return_meta_file(file):
    
    if file=="20211108-055826-CC-09-28_TV_102021":
        KEPT_TIME_INTERVAL=("2021-11-05 10:08:35","2021-11-06 00:45:06")
        START=datetime.datetime.strptime("2021-11-05 09:03:24","%Y-%m-%d %H:%M:%S")  #cf excel Recap
    elif file=='20211114-120603-CC-09-28_TV_102021':
        KEPT_TIME_INTERVAL=("2021-11-11 09:57:50","2021-11-12 01:45:00") #AGM2_TV_AN_08122020
        START=datetime.datetime.strptime("2021-11-11 09:44:30","%Y-%m-%d %H:%M:%S")
    elif file=='20211122-104139-CC-09-28_TV_102021':
        KEPT_TIME_INTERVAL=("2021-11-19 09:29:01","2021-11-20 01:05:28") #AGM2_TV_AN_08122020
        START=datetime.datetime.strptime("2021-11-19 09:14:52","%Y-%m-%d %H:%M:%S")
    elif file=='20211107-072758-CC-09-29_TV_102021':
        KEPT_TIME_INTERVAL=("2021-11-05 10:50:22","2021-11-05 15:33:29") #AGM2_TV_AN_08122020
        START=datetime.datetime.strptime("2021-11-05 09:03:55","%Y-%m-%d %H:%M:%S")
    elif file=='20220430-135054-CC-09-72':
        KEPT_TIME_INTERVAL=("2022-04-11 11:09:25","2022-04-11 23:12:30") #AGM2_TV_AN_08122020
        START=datetime.datetime.strptime("2022-04-11 11:05:13","%Y-%m-%d %H:%M:%S")
    elif file=='20220508-195728-CC-09-80':
        KEPT_TIME_INTERVAL=("2022-05-05 10:36:35","2022-05-05 13:56:02") #AGM2_TV_AN_08122020
        START=datetime.datetime.strptime("2022-05-05 10:33:32","%Y-%m-%d %H:%M:%S")
    elif file=='20211024-141045-CC-09-29_TV_102021':
        KEPT_TIME_INTERVAL=("2021-10-22 10:19:10","2021-10-23 00:38:00")
        START=datetime.datetime.strptime("2021-10-22 09:25:52","%Y-%m-%d %H:%M:%S")
    elif file=='20211101-081704-CC-09-28_TV_102021':
        KEPT_TIME_INTERVAL=("2021-10-29 08:50:13","2021-10-29 22:14:00")
        START=datetime.datetime.strptime("2021-10-29 08:41:01","%Y-%m-%d %H:%M:%S")
    elif file=='20211102-115134-CC-09-29_TV_102021':
        KEPT_TIME_INTERVAL=("2021-10-29 09:48:42","2021-10-29 19:19:00")
        START=datetime.datetime.strptime("2021-10-29 09:31:00","%Y-%m-%d %H:%M:%S")
    elif file=='20211113-213025-CC-09-29_TV_112021':
        KEPT_TIME_INTERVAL=("2021-11-11 10:20:49","2021-11-11 15:48:00")
        START=datetime.datetime.strptime("2021-11-11 10:03:05","%Y-%m-%d %H:%M:%S")
    elif file=='20211121-203649-CC-09-29_TV_112021':
        KEPT_TIME_INTERVAL=("2021-11-19 09:55:25","2021-11-20 01:19:00")
        START=datetime.datetime.strptime("2021-11-19 09:14:40","%Y-%m-%d %H:%M:%S")
    elif file=='20211130-203207-CC-09-29_TV_112021':
        KEPT_TIME_INTERVAL=("2021-11-28 07:45:15","2021-11-29 02:24:20")
        START=datetime.datetime.strptime("2021-11-28 07:34:58","%Y-%m-%d %H:%M:%S")
    elif file=='20220114-125839-CC-09-29_TV_012022':
        KEPT_TIME_INTERVAL=("2022-01-12 10:17:21","2022-01-13 00:38:00")
        START=datetime.datetime.strptime("2022-01-12 10:04:55","%Y-%m-%d %H:%M:%S")
    elif file=='20220414-002025-CC-09-80':
        KEPT_TIME_INTERVAL=("2022-04-11 12:11:35","2022-04-12 03:24:00")
        START=datetime.datetime.strptime("2022-04-11 12:06:33","%Y-%m-%d %H:%M:%S")
    elif file=='20220421-211441-CC-09-80':
        KEPT_TIME_INTERVAL=("2022-04-19 09:50:24","2022-04-19 23:56:00")
        START=datetime.datetime.strptime("2022-04-19 09:22:50","%Y-%m-%d %H:%M:%S")

    return KEPT_TIME_INTERVAL, START