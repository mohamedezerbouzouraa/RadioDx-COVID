import torch
from src.visualize import show_preds

def train(model, dl_train, dl_test, loss_fn, optimizer, test_dataset,
          class_names, epochs=1, eval_every=20, accuracy_threshold=0.95):
    for e in range(0, epochs):
        print('=' * 20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('=' * 20)
        train_loss = 0.
        val_loss = 0.
        model.train() 
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_step % eval_every == 0:
                print('Evaluating at step', train_step)
                accuracy = 0
                model.eval()  
                for val_step, (images, labels) in enumerate(dl_test):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    accuracy += sum((preds == labels).numpy())
                val_loss /= (val_step + 1)
                accuracy = accuracy / len(test_dataset)
                print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
                show_preds(model, dl_test, class_names)
                model.train()

                if accuracy >= accuracy_threshold:
                    print('Performance condition satisfied, stopping..')
                    return

        train_loss /= (train_step + 1)
        print(f'Training Loss: {train_loss:.4f}')
    print('Training complete..')
