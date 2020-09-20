from keras_contrib.layers import crf
import keras
class Compilemodel():
    def compilemodel(self,myobj):
        opt = keras.optimizers.Adam(lr=0.01)
        myobj.model_.compile(optimizer=opt, loss=myobj.crf.loss_function, metrics=[myobj.crf.accuracy])

        myobj.model_.summary()
        freport = open("logs/report.txt", "a", encoding="utf8")
        freport.write("model_.summary---------" + str(myobj.model_.summary()) + "\n")

        return myobj.model_
