/**
 *
 * @author adines
 */
public class Principal {

    public static void main(String[] args) {
        if (args.length == 2) {
            String image = args[0];
            String modelName = args[1];
            PredictorFactory predictorFactory=new PredictorFactory();
            Predictor predictor=predictorFactory.getPredictor("DL4J", modelName);
            System.out.println(predictor.predict(image));
        }
    }
}