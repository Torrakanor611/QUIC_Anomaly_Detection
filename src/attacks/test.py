#!/usr/bin/python

import requests as req

resp = req.get("https://192.168.56.101", verify='../../192.168.56.101.crt')

print(resp.text)