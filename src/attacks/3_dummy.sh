source ../venv/bin/activate

for (( ; ; ))
do
    python3 http3_client.py --ca-certs ../../testwebsite.crt https://192.168.56.101 > /dev/null 2>&1

done