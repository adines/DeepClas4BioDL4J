import java.util.function.Function;
import java.util.function.Supplier;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 *
 * @author adines
 */
public class Model {
    
    private org.deeplearning4j.nn.api.Model deepModel;
    private Function<String,INDArray> preProcessor;
    private Function<INDArray,String> postProcessor;
    
    public Model(Supplier<org.deeplearning4j.nn.api.Model> loadModel, Function<String, INDArray> preProcessor, Function<INDArray, String> postProcessor) {
        this.deepModel = loadModel.get();
        this.preProcessor = preProcessor;
        this.postProcessor = postProcessor;
    }

    public org.deeplearning4j.nn.api.Model getDeepModel() {
        return deepModel;
    }

    public Function<String, INDArray> getPreProcessor() {
        return preProcessor;
    }

    public Function<INDArray, String> getPostProcessor() {
        return postProcessor;
    }
    
    
}
