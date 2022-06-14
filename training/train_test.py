import torch
import numpy as np
from tqdm import tqdm
from utils.utils import f1_score


def pytorch_training_loop(model, optimizer, scheduler, train_data, val_data, num_labels, start_epoch, total_epochs,
                          batch_size, model_save_path, best_val_score=0, patience=5, shuffle_data=False):

    epoch = start_epoch
    n_train = train_data.num_examples
   
    non_improvement = 0
    
    for epoch in range(epoch, total_epochs):
            
        epoch_loss = 0
        epoch_p = 0
        epoch_r = 0
        epoch_f = 0
        epoch_test_steps = 0

        print("Training epoch {}".format(epoch+1))
        if non_improvement > 0:
            print("Early stopping {}/{}".format(non_improvement, patience))

        model.train()
        
        if shuffle_data:
            train_data.shuffle(random_seed=epoch)

        pbar = tqdm(range(0, n_train, batch_size), unit="steps", desc="Training")

        # Training loop
        for i in pbar:
            end_ix = min(i+batch_size, n_train)
            start_ix = end_ix - batch_size
            step_count = int(start_ix/batch_size)

            input_dict = train_data.get_batch(start_ix, end_ix, return_labels=True)
            
            optimizer.zero_grad()
            
            model_output = model(**input_dict)
            output = model_output['output']
            loss = model_output['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            if step_count % 10 == 0:
                p, r, f = f1_score(output.cpu().detach().numpy(),
                                   input_dict['target'].cpu().detach().numpy(), 
                                   0.5, average='micro')
                epoch_test_steps += 1
                epoch_loss += loss.detach().item()
                epoch_p += p
                epoch_r += r
                epoch_f += f
                pbar.set_postfix_str(
                    "Loss {:.4f}, Precision {:.4f}, Recall {:.4f}, F1 {:.4f}".format(
                        epoch_loss/epoch_test_steps,
                        epoch_p/epoch_test_steps,
                        epoch_r/epoch_test_steps,
                        epoch_f/epoch_test_steps
                    )
                )

        # Validation loop
        val_probs, val_targs = pytorch_testing_loop(model, val_data, num_labels, batch_size)

        val_p, val_r, val_f = f1_score(val_probs, val_targs, 0.5, average='micro')
        if val_f >= best_val_score:
            best_val_score = val_f
            non_improvement = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            non_improvement += 1
        print('Precision (val): {:.4f}, Recall (val): {:.4f}, F1 (val): {:.4f} | Best validation F1: {:.4f}\n'.format(
            val_p, val_r, val_f, best_val_score)
        )
        
        if non_improvement == patience:
            return best_val_score

        scheduler.step(val_f)
        
    return best_val_score


def pytorch_testing_loop(model, test_data, num_labels,
                         batch_size):

    n_test = test_data.num_examples
    
    # Testing loop
    test_probs = np.zeros((n_test, num_labels))
    test_targs = np.zeros((n_test, num_labels))
    model.eval()

    pbar = tqdm(range(0, n_test, batch_size), unit="steps", desc="Testing")
    for i in pbar:
        end_ix = min(i+batch_size, n_test)
        start_ix = end_ix - batch_size

        input_dict = test_data.get_batch(start_ix, end_ix, return_labels=True)
        
        with torch.no_grad():
            model_output = model(**input_dict)
            output = model_output['output']
                    
        _probs = output.cpu().detach().numpy()
        test_probs[start_ix:end_ix, :] = _probs
        test_targs[start_ix:end_ix, :] = input_dict['target'].cpu().detach().numpy()

    return test_probs, test_targs
