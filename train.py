from models import feedforward


def train_feedforward(x,y,hp):
    model = feedforward.make_model(hp);
    model.fit(x,y,validation_split=0.2,batch_size=hp['batch_size'], epochs=hp['epochs'])
