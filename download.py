from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('baichuan-inc/Baichuan2-7B-Chat', cache_dir='./autodl-tmp/Baichuan2-7B-Chat', revision='master')
