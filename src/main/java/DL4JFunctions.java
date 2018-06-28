import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.GoogLeNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.deeplearning4j.zoo.model.VGG19;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

/**
 *
 * @author adines
 */
public class DL4JFunctions {
    
    
     /*******************************
     * METHODS FOR LOAD MODELS
     *******************************/
    
    public static Model loadModel(String model){
        Model m=null;
        try {
            m= ModelSerializer.restoreComputationGraph("DL4J/Classification/model/"+model+".zip");
        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            return m;
        }
    }
    
    public static Model vgg16dl4jload()
    {
        Model m=null;
        try {
            ZooModel zooModel = new VGG16();
            m=zooModel.initPretrained(PretrainedType.IMAGENET);
        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            return m;
        }
    }
    
    public static Model vgg19dl4jload()
    {
        Model m=null;
        try {
            ZooModel zooModel = new VGG19();
            m=zooModel.initPretrained(PretrainedType.IMAGENET);
        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            return m;
        }
    }
    
    public static Model resnet50dl4jload()
    {
        Model m=null;
        try {
            ZooModel zooModel = new ResNet50();
            m= zooModel.initPretrained(PretrainedType.IMAGENET);
        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }catch (Exception ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }
        finally{
            return m;
        }
    }
    
    public static Model googlenetdl4jload()
    {
        Model m=null;
        try {
            ZooModel zooModel = new GoogLeNet();
            m= zooModel.initPretrained(PretrainedType.IMAGENET);
        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }catch (Exception ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }
        finally{
            return m;
        }
    }
    
    
    
    /*******************************
     * METHODS FOR PREPROCESS
     *******************************/
    
    public static INDArray vgg16dl4jpreprocess(String image)
    {
        INDArray input=null;
        try {
            
            NativeImageLoader imLoader=new NativeImageLoader(224, 224, 3);
            input=imLoader.asMatrix(new File(image));
            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(input);

        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            return input;
        }
    }
    
    public static INDArray vgg19dl4jpreprocess(String image)
    {
        INDArray input=null;
        try {
            
            NativeImageLoader imLoader=new NativeImageLoader(224, 224, 3);
            input=imLoader.asMatrix(new File(image));
            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(input);

        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            return input;
        }
    }
    
    public static INDArray resnet50dl4jpreprocess(String image)
    {
        INDArray input=null;
        try {
            NativeImageLoader imLoader=new NativeImageLoader(224, 224, 3);
            input=imLoader.asMatrix(new File(image));
        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            return input;
        }
    }
    
    public static INDArray googlenetdl4jpreprocess(String image)
    {
        INDArray input=null;
        try {
            NativeImageLoader imLoader=new NativeImageLoader(224, 224, 3);
            input=imLoader.asMatrix(new File(image));
        } catch (IOException ex) {
            Logger.getLogger(DL4JFunctions.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            return input;
        }
    }
    
    
    /*******************************
     * METHODS FOR POSTPROCESS
     *******************************/
    
    public static String vgg16dl4jpostprocess(INDArray prediction)
    {
        ImageNetLabels il=new ImageNetLabels();
        String s=il.decodePredictions(prediction);
        int i=s.indexOf("%, ");
        return s.substring(i+3, s.indexOf("\n", i));
    }
    
    public static String vgg19dl4jpostprocess(INDArray prediction)
    {
        ImageNetLabels il=new ImageNetLabels();
        String s=il.decodePredictions(prediction);
        int i=s.indexOf("%, ");
        return s.substring(i+3, s.indexOf("\n", i));
    }
    
    public static String resnet50dl4jpostprocess(INDArray prediction)
    {
        ImageNetLabels il=new ImageNetLabels();
        String s=il.decodePredictions(prediction);
        int i=s.indexOf("%, ");
        return s.substring(i+3, s.indexOf("\n", i));
    }
    
    public static String googlenetdl4jpostprocess(INDArray prediction)
    {
        ImageNetLabels il=new ImageNetLabels();
        String s=il.decodePredictions(prediction);
        int i=s.indexOf("%, ");
        return s.substring(i+3, s.indexOf("\n", i));
    }
}
