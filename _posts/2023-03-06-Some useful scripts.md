# some useful scripts

## images train val test split

```python
import splitfolders

def train_test_split(in_path,out_path):
    splitfolders.ratio(in_path, output=out_path, seed=1337, ratio=(.8, .0, .2), group_prefix=None, move=False)

input_path =r"\data"
output_path = r"\data_splitted"
train_test_split(input_path,output_path)
```

## loss accuracy plots

```python
def loss_acc_plot(train_losses,val_losses,train_acc,val_acc,name):
    plt.plot(range(len(train_losses)),train_losses,'b')
    plt.plot(range(len(val_losses)),val_losses,'r')
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('loss vs number of epochs')
    plt.legend(['train','val'])
    plt.savefig(rf'fig\loss_{name}.png')
    plt.show()

    plt.plot(range(len(train_acc)),train_acc,'b')
    plt.plot(range(len(val_acc)),val_acc,'r')
    plt.xlabel('number of epochs')
    plt.ylabel('acc')
    plt.title('acc vs number of epochs')
    plt.legend(['train','val'])
    plt.savefig(rf'fig\acc_{name}.png')
    plt.show()
```

## cls task train.py

```python
def train(num_epochs, train_loader,val_loader,model,optimizer,scheduler,criterion, model_date_name,device):
    train_epoch_losses = []
    val_epoch_losses = []
    train_epoch_acc=[]
    val_epoch_acc=[]
    for epoch in range(num_epochs):
        train_epoch_loss = 0.0
        model.train()
        correct = 0.0
        total = 0.0
        for i, (data, label) in enumerate(train_loader):
            data, label = data.to(device),label.to(device)
            optimizer.zero_grad()
            #forward-pass
            output = model(data)
            #batch loss
            loss = criterion(output, label)
            loss.backward()
            # Update the parameters
            optimizer.step()
            train_epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += label.size(0)
            #eq 比较各项是否相同 同为true 否为false sum以true为1相加 得到correct的个数
            correct += predicted.eq(label).sum().item()
            #print(f'iter:{i} done, total{len(train_loader)}')
        print(f'current epoch:{epoch} train loss:{train_epoch_loss}')
        print(f'current epoch:{epoch} train acc:{(100. * correct / total)}')
        train_epoch_losses.append(train_epoch_loss)
        train_epoch_acc.append((100. * correct / total))
        model.eval()
        val_epoch_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for j, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_epoch_loss += loss.item()
                #val_losses.append(loss.item())
                _, predicted = outputs.max(1)
                #print(outputs.data)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f'current epoch:{epoch} val loss:{val_epoch_loss}')
        print(f'current epoch:{epoch} val acc:{(100. * correct / total)}')
        val_epoch_losses.append(val_epoch_loss)
        val_epoch_acc.append((100. * correct / total))
        scheduler.step()
        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), rf'C:\Users\raych\github_project\mine\usqualitycls\checkpoint\{model_date_name}_{epoch+1}.pt')

    torch.save(model.state_dict(), rf'C:\Users\raych\github_project\mine\usqualitycls\checkpoint\{model_date_name}_final.pt')
    return train_epoch_losses, val_epoch_losses, train_epoch_acc, val_epoch_acc, model
```

## inference video\folder\image

```python
def inference(in_path,out_path,categories,model):
    model.eval()
    cap = cv2.VideoCapture(in_path)
    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # define the outfile file name
    
    #save_name = f"{in_path.split('/')[-1].split('.')[0]}_{DEVICE}"
    # define codec and create VideoWriter object 
    out = cv2.VideoWriter(out_path, 
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        (frame_width, frame_height))
    # to count the total number of frames iterated through
    frame_count = 0
    # to keep adding the frames' FPS
    total_fps = 0
    while(cap.isOpened()):
        # capture each frame of the video
        start_time = time.time()
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # apply transforms to the input image
            input_tensor = transform(rgb_frame)
            # add the batch dimensionsion
            input_batch = input_tensor.unsqueeze(0) 

            # move the input tensor and resnet18_us to the computation device
            input_batch = input_batch.to(DEVICE)
            model.to(DEVICE)

            with torch.no_grad():
                output = model(input_batch)
            # get the softmax probabilities of dim1 
            probabilities = torch.nn.functional.softmax(output,1)
            _,index = torch.max(probabilities,1)
            # get the current fps
            end_time = time.time()+0.000001
            fps = 1 / (end_time - start_time)
            # # add `fps` to `total_fps`
            total_fps += fps
            # increment frame count
            frame_count += 1
            cv2.putText(frame, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            cv2.putText(frame, f"{categories[index.tolist()[0]]}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            cv2.imshow('Result', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    #avg_fps = total_fps / frame_count
    #print(f"Average FPS: {avg_fps:.3f}")

def inference_folder(in_path,out_path,model,categories):
    model.eval()
    for file in os.listdir(in_path):
        fin_path = os.path.join(in_path,file)
        img = cv2.imread(fin_path)
        input_tensor = transform(img)
        # add the batch dimensionsion
        input_batch = input_tensor.unsqueeze(0) 
        input_batch = input_batch.to(DEVICE)
        model.to(DEVICE)

        with torch.no_grad():
            output = model(input_batch)
        # get the softmax probabilities of dim1 
        probabilities = torch.nn.functional.softmax(output,1)
        _,index = torch.max(probabilities,1)
        cv2.putText(img, f"{categories[index.tolist()[0]]}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
        fout_path = os.path.join(out_path,file)
        cv2.imwrite(fout_path,img)
        
def predict_one_image(image_path, model, transform, classes):
    unloader = transforms.ToTensor()
    test_img = Image.open(image_path).convert('RGB')
    plt.imshow(test_img)  # 展示预测的图片
    plt.show()
    test_img = unloader(test_img)
    test_img = transform(test_img)
    
    img = test_img.to(DEVICE).unsqueeze(0)
    
    model.eval()
    s_t = time.time()
    output = model(img)
    e_t = time.time()
    _,pred = torch.max(output,1)
    pred_class = classes[pred]
    print(f'预测结果是：{pred_class}')
    print(f"耗时:{s_t-e_t}")
```

