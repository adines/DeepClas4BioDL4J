
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 *
 * @author adines
 */
public interface IPredictor {
    public abstract String predict(String image);
    
    public abstract INDArray[] predictBatch(String []images);
    
}
