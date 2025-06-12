import torch

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, print_val_loss, train_log, args):
    # Training Loop
    for epoch in range(num_epochs):
        total_loss = 0.0 # To accumulate loss for each epoch
        model.train()
        for batch_X, batch_y in train_loader:
            # Move data to the appropriate device if necessary (e.g., GPU)
            # X: shape (b, n_steps, feat)
            # y: next step, shape (b)
            batch_X , batch_y = batch_X.to(args.DEVICE), batch_y.to(args.DEVICE)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)
        scheduler.step()

        if epoch % print_val_loss == 0 and epoch > 0:
            torch.save(model.state_dict(), args.data_dir + f'/best_{model.__class__.__name__}.pth')

            # Calculate average loss for the epoch
            epoch_loss = total_loss / len(train_loader.dataset)
            train_log.train_loss.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training_Loss: {epoch_loss:.6f}")

            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X , batch_y = batch_X.to(args.DEVICE), batch_y.to(args.DEVICE)
                    outputs, _ = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    total_loss += loss.item() * batch_X.size(0)
                epoch_loss = total_loss / len(val_loader.dataset)
                train_log.eval_loss.append(epoch_loss)
                print(f"Epoch [{epoch+1}/{num_epochs}], Eval_Loss: {epoch_loss:.3f}")