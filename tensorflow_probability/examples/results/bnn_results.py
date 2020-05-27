import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 


prefix = 'lr_1e-05'
# Load data
train_accuracy = np.load(prefix + 'train_accuracy.npy')
train_loss = np.load(prefix+'train_loss.npy')
test_accuracy = np.load(prefix+'test_accuracy.npy')
test_loss = np.load(prefix+'test_loss.npy')



# Train accuracy
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = train_accuracy.shape[0]
ax.plot(np.arange(x)+1, train_accuracy)
ax.set_title('Train accuracy')
ax.set_xlabel('Batches')
ax.set_ylabel('Accuracy (%)')
ax.legend(['Train accuracy'])
ax.grid(True)
plt.savefig('train_accuracy.png')

# Train loss
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = train_loss.shape[0]
ax.plot(np.arange(x)+1, train_loss)
ax.set_title('Train loss')
ax.set_xlabel('Batches')
ax.set_ylabel('Loss')
ax.legend(['Train loss'])
ax.grid(True)
plt.savefig('train_loss.png')

# Test accuracy
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = test_accuracy.shape[0]
ax.plot(np.arange(x)+1, test_accuracy, color='orange')
ax.set_title('Test accuracy')
ax.set_xlabel('Batches')
ax.set_ylabel('Accuracy (%)')
ax.legend(['Test accuracy'])
ax.grid(True)
plt.savefig('test_accuracy.png')

# Test loss
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = test_loss.shape[0]
ax.plot(np.arange(x)+1, test_loss, color='orange')
ax.set_title('Test loss')
ax.set_xlabel('Batches')
ax.set_ylabel('Loss')
ax.legend(['Test loss'])
ax.grid(True)
plt.savefig('test_loss.png')



# # subplots
# fig = plt.figure(figsize=(16, 8))
# ax0 = fig.add_subplot(1, 2, 1)
# ax1 = fig.add_subplot(1, 2, 2)
# x0_axis = np.arange(len(train_accuracy)) + 1
# ax0.plot(x0_axis, train_accuracy, x0_axis, test_accuracy)
# ax0.set_xlabel('Batches')
# ax0.set_ylabel('Accuracy (%)')
# ax0.legend(['Train accuracy', 'Test accuracy'])
# ax0.grid(True)

# x1_axis = np.arange(len(train_loss)) + 1
# ax1.plot(x1_axis, train_loss, x1_axis, test_loss)
# ax1.set_xlabel('Batches')
# ax1.set_ylabel('Loss')
# ax1.legend(['Train loss', 'Test loss'])

# plt.savefig('summary.png')

