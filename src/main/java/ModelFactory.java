import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 *
 * @author adines
 */
public class ModelFactory {
    
    public Model getModel(String framework, String model)
    {
        Model m=null;
        try {
            String className="DL4JFunctions";
            String nameMethodLoad=(model+"dl4j"+"load").toLowerCase();
            String nameMethodPreprocess=(model+"dl4j"+"preprocess").toLowerCase();
            String nameMethodPostprocess=(model+"dl4j"+"postprocess").toLowerCase();
            
            Method methodLoad=Class.forName(className).getDeclaredMethod(nameMethodLoad);
            Method methodPreprocess=Class.forName(className).getDeclaredMethod(nameMethodPreprocess,String.class);
            Method methodPostprocess=Class.forName(className).getDeclaredMethod(nameMethodPostprocess,INDArray.class);
            
            Supplier<org.deeplearning4j.nn.api.Model> loadModel=()->{
                org.deeplearning4j.nn.api.Model out=null;
                try {
                    out= (org.deeplearning4j.nn.api.Model)methodLoad.invoke(null);
                } catch (IllegalAccessException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IllegalArgumentException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                } catch (InvocationTargetException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                }finally{
                    return out;
                }
            };
            
            Function<String,INDArray> preProcessor=(s)->{
                INDArray out=null;
                try {
                    out=(INDArray)methodPreprocess.invoke(null, s);
                } catch (IllegalAccessException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IllegalArgumentException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                } catch (InvocationTargetException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                }finally{
                    return out;
                }
            };
            
            Function<INDArray,String> postPorcessor=(i)->{
                String out=null;
                try {
                    out=(String)methodPostprocess.invoke(null, i);
                } catch (IllegalAccessException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                } catch (IllegalArgumentException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                } catch (InvocationTargetException ex) {
                    Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
                }finally{
                    return out;
                }
            };
            
            m=new Model(loadModel, preProcessor, postPorcessor);
            
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
        } catch (NoSuchMethodException ex) {
            Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
        } catch (SecurityException ex) {
            Logger.getLogger(ModelFactory.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            return m;
        }
    }
}
