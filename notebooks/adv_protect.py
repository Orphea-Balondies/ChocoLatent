import os
import subprocess
import json
import torch 
import concurrent.futures
import multiprocessing as mp

init_path='init_images'
model_path='stable-diffusion-v1-5'
image_list=[]
for dir in ['mix-27']:
  for _,_,files in os.walk(os.path.join(init_path,dir)):
    for f in files:
      if f.split('.')[-1].lower() not in ['jpg','png','jpeg']:
        continue
      file_path = os.path.join(dir,f)
      for out_prefix in ['10ScimP01aprBlack01E','40ScimP01aprBlack01E','70ScimP01aprBlack01E','100ScimP01aprBlack01E']:
        out_file = f.split('.')[0] + f'_adv-photoguard-{out_prefix}.png'
        out_true_path = os.path.join(init_path,model_path,out_prefix,dir,out_file)
        if not os.path.exists(out_true_path):
            image_dict={"image_name": file_path}
            image_list.append(image_dict)
            break
        
num_gpus = torch.cuda.device_count()
def chunk_data(data_list, num_chunks):
    return [data_list[i::num_chunks] for i in range(num_chunks)]
data_chunks = chunk_data(image_list, num_gpus)


def process_data_on_gpu(data_chunk, device_id):
    device = torch.device(f'cuda:{device_id}')
    print(f"Processing on GPU {device_id}")
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for data in [data_chunk[i::3] for i in range(3)]:
                future = executor.submit(run_worker_script, json.dumps(data), device, '--dir', init_path, '--mode', 'protect', '--out_dir', f'{init_path}/{model_path}', '--model_path', 'stable-diffusion-xl-base-1.0', '--SCIMvalues', '10', '40', '70', '100')
                futures.append(future)
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Error during task execution: {e}")

    except Exception as e:
        print(f"Error in process_data_on_gpu: {e}")

    
def parallel_processing(data_chunks, num_gpus):
    processes = []

    with mp.Manager() as manager:

        for i in range(num_gpus):
            p = mp.Process(target=process_data_on_gpu, args=(data_chunks[i], i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

# 并行处理数据


def run_worker_script( data, device_id, *args):
    # 构建要执行的命令
    print(f"Processing {data} on GPU {device_id}")
    device = f'{device_id}'
    
    args = ['--image_list', data ,'--device', device] + list(args)
    command = ['python', 'notebooks/demo_simple_attack_img2img.py'] + args
    try:
    # 使用 subprocess 运行命令
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        print(f"Error in process_data_on_gpu: {e}")
        return

    while True:
        output = process.stdout.readline()
        error_output = process.stderr.readline()
        if output == '' and error_output == '' and process.poll() is not None:
            break
            
        if output:
            print(output.strip())  # 正常输出
        
        if error_output:
            print(error_output.strip()) 
    rc = process.poll()
    return

if __name__ == "__main__":
    parallel_processing(data_chunks, num_gpus)