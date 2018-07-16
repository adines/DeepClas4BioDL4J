
import java.util.Arrays;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author adines
 */
public class Principal {

    public static void main(String[] args) {
        if (args.length == 2) {
            String image = args[1];
            String modelName = args[0];
            PredictorFactory predictorFactory=new PredictorFactory();
            Predictor predictor=predictorFactory.getPredictor("DL4J", modelName);
            System.out.println(predictor.predict(image));
        }
        else if (args.length > 2) {
            String modelName = args[0];
            String images[]=Arrays.copyOfRange(args,1,args.length);
            PredictorFactory predictorFactory=new PredictorFactory();
            Predictor predictor=predictorFactory.getPredictor("DL4J", modelName);
            for(INDArray i : predictor.predictBatch(images))
            {
                System.out.println(i);
            }
        }
    }
}