import csv
import torch


def inference(data_loader, model, result_path):
    print('inference')
    header = ['target', 'output']
    with open(result_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        model.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(data_loader):

                outputs = model(inputs)

                for j in range(outputs.size(0)):
                    writer.writerow([targets[j],outputs[j]])

                print('[{}/{}]\t'.format(i + 1, len(data_loader)))