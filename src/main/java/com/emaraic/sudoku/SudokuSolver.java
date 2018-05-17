package com.emaraic.sudoku;

import com.emaraic.utils.LinesComparator;
import com.emaraic.utils.Sudoku;
import java.awt.BorderLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.SwingUtilities;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.opencv_core.RotatedRect;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_core.CV_8UC3;
import static org.bytedeco.javacpp.opencv_core.CV_PI;
import org.bytedeco.javacpp.opencv_core.CvPoint;
import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_COMPLEX;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Point2f;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_core.bitwise_not;
import static org.bytedeco.javacpp.opencv_core.copyMakeBorder;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.ADAPTIVE_THRESH_GAUSSIAN_C;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_GRAY2BGR;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;
import static org.bytedeco.javacpp.opencv_imgproc.CV_FILLED;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_TREE;
import static org.bytedeco.javacpp.opencv_imgproc.Canny;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
import static org.bytedeco.javacpp.opencv_imgproc.HoughLines;
import static org.bytedeco.javacpp.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.THRESH_BINARY_INV;
import static org.bytedeco.javacpp.opencv_imgproc.WARP_INVERSE_MAP;
import static org.bytedeco.javacpp.opencv_imgproc.adaptiveThreshold;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;
import static org.bytedeco.javacpp.opencv_imgproc.circle;
import static org.bytedeco.javacpp.opencv_imgproc.contourArea;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.findContours;
import static org.bytedeco.javacpp.opencv_imgproc.getPerspectiveTransform;
import static org.bytedeco.javacpp.opencv_imgproc.getRotationMatrix2D;
import static org.bytedeco.javacpp.opencv_imgproc.line;
import static org.bytedeco.javacpp.opencv_imgproc.minAreaRect;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.warpAffine;
import static org.bytedeco.javacpp.opencv_imgproc.warpPerspective;
import static org.bytedeco.javacpp.opencv_videoio.CV_CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.javacpp.opencv_videoio.CV_CAP_PROP_FRAME_WIDTH;
import org.bytedeco.javacpp.opencv_videoio.VideoCapture;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.LoggerFactory;

/**
 * Kindly, Don't Remove this Header.
 *
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: May 10, 2018
 */
public class SudokuSolver {

    private static final OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
    private final static int[] DIGITS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    private static MultiLayerNetwork NETWORK;
    private static final org.slf4j.Logger log = LoggerFactory.getLogger(SudokuSolver.class);

    private static MultiLayerNetwork loadNetwork() {
        MultiLayerNetwork network = null;
        try {
            String pathtoexe = System.getProperty("user.dir");
            File net = new File(pathtoexe, "cnn-model.data");
            network = ModelSerializer.restoreMultiLayerNetwork(net);
        } catch (IOException ex) {
            log.error("Error While Loading Pretrained Network: " + ex.getMessage());
        }
        return network;
    }

    /*Check if the captured image contains sudoku pazzle 
    assume that it has a large square with area > 40000*/
    private static boolean isSudokuExist(Mat img) {
        MatVector countours = new MatVector();
        List<Double> araes = new ArrayList<>();
        findContours(img.clone(), countours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, new Point(0, 0));
        for (int i = 0; i < countours.size(); i++) {
            Mat c = countours.get(i);
            double area = contourArea(c);
            araes.add(area);
        }
        if (araes.isEmpty()) {
            return false;
        }
        Double d = Collections.max(araes);
        return d > 40000;
    }

    /*Get the largest Rectangle of an image*/
    private static Rect getLargestRect(Mat img) {
        MatVector countours = new MatVector();
        List<Rect> rects = new ArrayList<>();
        List<Double> araes = new ArrayList<>();
        findContours(img, countours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, new Point(0, 0));
        for (int i = 0; i < countours.size(); i++) {
            Mat c = countours.get(i);
            double area = contourArea(c);
            Rect boundingRect = boundingRect(c);
            araes.add(area);
            rects.add(boundingRect);
        }
        if (araes.isEmpty() || Collections.max(araes) < 4000) {
            return new Rect(0, 0, img.cols(), img.rows());
        } else {
            Double d = Collections.max(araes);
            return rects.get(araes.indexOf(d));
        }
    }

    /*Print corners points of detected rectangle*/
    private static void printCornerPoints(Rect r, Mat colorimg) {
        Point tl = new Point(r.x(), r.y());
        Point dl = new Point(r.x(), r.y() + r.height());
        Point tr = new Point(r.x() + r.width(), r.y());
        Point dr = new Point(r.x() + r.width(), r.y() + r.height());
        circle(colorimg, tl, 10, new Scalar(255, 255, 0, 255), CV_FILLED, 8, 0);
        circle(colorimg, dl, 10, new Scalar(255, 255, 0, 255), CV_FILLED, 8, 0);
        circle(colorimg, tr, 10, new Scalar(255, 255, 0, 255), CV_FILLED, 8, 0);
        circle(colorimg, dr, 10, new Scalar(255, 255, 0, 255), CV_FILLED, 8, 0);
    }

    /* warpPrespectivePuzzle && deskewImage are responsible for correct(deskew) rotated image */
    private static Mat warpPrespectivePuzzle(Mat image) {
        image = deskewImage(image);
        Rect rect = getLargestRect(image);
        Point2f srcPts = new Point2f(4);
        srcPts.position(0).x((float) rect.x()).y((float) rect.y());
        srcPts.position(1).x((float) rect.x() + rect.width()).y((float) rect.y());
        srcPts.position(2).x((float) rect.x() + rect.width()).y((float) rect.y() + rect.height());
        srcPts.position(3).x((float) rect.x()).y((float) rect.y() + rect.height());

        Point2f dstPts = new Point2f(4);
        dstPts.position(0).x(0).y(0);
        dstPts.position(1).x(600 - 2).y(0);
        dstPts.position(2).x(600 - 2).y(600 - 2);
        dstPts.position(3).x(0).y(600 - 2);

        Mat p = getPerspectiveTransform(srcPts.position(0), dstPts.position(0));
        Mat img = new Mat(new Size(600, 600), image.type());//image.size()
        warpPerspective(image, img, p, img.size());
        return img;
    }

    private static Mat deskewImage(Mat img) {
        MatVector countours = new MatVector();
        List<Double> araes = new ArrayList<>();
        findContours(img, countours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, new Point(0, 0));
        for (int i = 0; i < countours.size(); i++) {
            Mat c = countours.get(i);
            double area = contourArea(c);
            araes.add(area);
        }
        if (araes.isEmpty()) {
            return img;
        } else {
            Double d = Collections.max(araes);
            RotatedRect minAreaRect = minAreaRect(countours.get(araes.indexOf(d)));
            float angle = minAreaRect.angle();
            if (angle < -45) {
                angle = -(90 + angle);
            } else {
                angle = -angle;
            }
            Mat rot = getRotationMatrix2D(minAreaRect.center(), angle, 1);
            Mat dst = new Mat(img.size(), img.type());
            warpAffine(img, dst, rot, dst.size(), WARP_INVERSE_MAP | INTER_LINEAR, BORDER_CONSTANT, new Scalar(0, 0, 0, 0));
            return dst;
        }
    }

    /*Check that the distance between any two consecutive lines must be larger than a value (40) to get rid of crossed lines*/
    private static boolean checkLines(List<Cluster> vlines, List<Cluster> hlines) {
        final int diff = 40;//this may vary if you change the image width and hieght in method warpPrespectivePuzzle (600)
        if (!(vlines.size() == 10 && hlines.size() == 10)) {
            return false;
        }
        for (int i = 0; i < hlines.size() - 1; i++) {
            Cluster get = hlines.get(i);
            double r1 = get.getCenter().getArray().getDouble(0);
            Cluster get1 = hlines.get(i + 1);
            double r2 = get1.getCenter().getArray().getDouble(0);
            if (Math.abs(r1 - r2) < diff) {
                return false;
            }
        }
        for (int i = 0; i < vlines.size() - 1; i++) {
            Cluster get = vlines.get(i);
            double r1 = get.getCenter().getArray().getDouble(0);
            Cluster get1 = vlines.get(i + 1);
            double r2 = get1.getCenter().getArray().getDouble(0);
            if (Math.abs(r1 - r2) < diff) {
                return false;
            }
        }
        return true;
    }

    /*Get points of intersection between vertical and horizontal lines*/
    private static List<Point> getPoint(List<Cluster> vlines, List<Cluster> hlines) {
        List<Point> points = new ArrayList();
        for (int i = 0; i < hlines.size(); i++) {
            Cluster get = hlines.get(i);
            double r1 = get.getCenter().getArray().getDouble(0);
            double t1 = get.getCenter().getArray().getDouble(1);
            for (int j = 0; j < vlines.size(); j++) {
                Cluster get1 = vlines.get(j);
                double r2 = get1.getCenter().getArray().getDouble(0);
                double t2 = get1.getCenter().getArray().getDouble(1);
                Point o = parametricIntersect(r1, t1, r2, t2);
                if (o.y() != -1 & o.x() != -1) {
                    points.add(o);
                }
            }
        }
        for (int i = 0; i < points.size() - 1; i++) {
            Point get = points.get(i);
            Point get1 = points.get(i + 1);
            if (getDistance(get, get1) < 20) {
                points.remove(get);
            }
        }
        //System.out.println("Points Size" + points.size());
        return points;
    }

    /*get intersection points between two lines given their rhoes and thetas*/
    private static Point parametricIntersect(Double r1, Double t1, Double r2, Double t2) {
        double ct1 = Math.cos(t1);     //matrix element a
        double st1 = Math.sin(t1);     //b
        double ct2 = Math.cos(t2);     //c
        double st2 = Math.sin(t2);     //d
        double d = ct1 * st2 - st1 * ct2;//determinative (rearranged matrix for inverse)
        if (d != 0.0f) {
            int x = (int) ((st2 * r1 - st1 * r2) / d);
            int y = (int) ((-ct2 * r1 + ct1 * r2) / d);
            return new Point(x, y);
        } else { //lines are parallel and will NEVER intersect!
            return new Point(-1, -1);
        }
    }

    static double getDistance(Point p1, Point p2) {
        return Math.sqrt(Math.pow((p1.x() - p2.x()), 2) + Math.pow((p1.y() - p2.y()), 2));
    }


    /*Dected a digit given a cell mat object*/
    private static Mat detectDigit(Mat img) {
        Mat res = new Mat();
        MatVector countours = new MatVector();
        List<Rect> rects = new ArrayList<>();
        List<Double> araes = new ArrayList<>();
        bitwise_not(img, img);
        findContours(img, countours, opencv_imgproc.CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, new Point(0, 0));
        for (int i = 0; i < countours.size(); i++) {
            Mat c = countours.get(i);
            Rect boundbox = boundingRect(c);
            if (boundbox.height() > 20 && boundbox.height() < 50 && boundbox.width() > 15 && boundbox.width() < 40) {
                double aspectRatio = boundbox.height() / boundbox.width();
                //System.out.println("Aspect ratio " + aspectRatio);
                if (aspectRatio >= 1 && aspectRatio < 3) {
                    rects.add(boundbox);
                    double area = contourArea(c);
                    araes.add(area);
                }
            }
        }
        if (!araes.isEmpty()) {
            bitwise_not(img, img);

            Double d = Collections.max(araes);
            res = img.apply(rects.get(araes.indexOf(d)));
            copyMakeBorder(res, res, 10, 10, 10, 10, BORDER_CONSTANT, new Scalar(255, 255, 255, 255));
            resize(res, res, new Size(28, 28));
            return res;
        } else {
            return img;//org.bytedeco.javacpp.helper.AbstractMat.EMPTY
        }
    }

    /*Recognise digit given its image*/
    private static int recogniseDigit(Mat digit) {
        int idx = 0;
        try {
            NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
            bitwise_not(digit, digit);
            INDArray dig = loader.asMatrix(digit);
            INDArray flaten = dig.reshape(new int[]{1, 784});
            flaten = flaten.div(255);
            INDArray output = NETWORK.output(flaten);
            idx = Nd4j.getExecutioner().execAndReturn(new IAMax(output)).getFinalResult();
            //imwrite("di/" + i + ".jpg", digit);
            digit.release();
        } catch (IOException ex) {
            log.error(ex.getMessage());
        }
        return DIGITS[idx];
    }
    /*Print th results of sudoku to a given matrix*/
    private static void printResult(Mat img, INDArray result, INDArray puzzle, List<Rect> rects) {
        for (int i = 0; i < rects.size(); i++) {
            Rect rect = rects.get(i);
            int x = rect.x();
            int y = rect.y();
            int d = (int) result.getDouble(i / 9, i % 9);
            int d1 = (int) puzzle.getDouble(i / 9, i % 9);
            if (d != d1) {//Print Solution

                putText(img, d + "", new Point(x + 20, y + 50),
                        FONT_HERSHEY_COMPLEX, 1.3, new Scalar(255, 0, 0, 0), 3, 2, false);
            } else {//Print Recognised Puzzle

                putText(img, d + "", new Point(x + 10, y + 40),
                        FONT_HERSHEY_COMPLEX, 1, new Scalar(0, 0, 255, 0), 2, 2, false);
            }

        }

    }
    /*Check that a INDarray contains zeros, to validate sudoku solution*/
    private static boolean isContainsZero(INDArray puz) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (puz.getInt(i, j) == 0) {
                    return true;
                }
            }
        }
        return false;
    }

    public static void main(String[] args) {
        /*Load Pre-trained Network */
        NETWORK = loadNetwork();

        final AtomicReference<VideoCapture> capture = new AtomicReference<>(new VideoCapture());
        capture.get().set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        capture.get().set(CV_CAP_PROP_FRAME_HEIGHT, 720);

        if (!capture.get().open(0)) {
            log.error("Can not open the cam !!!");
        }

        final AtomicReference<Boolean> start = new AtomicReference<>(true);

        Mat colorimg = new Mat();

        CanvasFrame mainframe = new CanvasFrame("Real-time Sudoku Solver - Emaraic");
        mainframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        mainframe.setCanvasSize(600, 600);
        mainframe.setLocationRelativeTo(null);
        mainframe.setLayout(new BoxLayout(mainframe.getContentPane(), BoxLayout.Y_AXIS));
        JButton control = new JButton("Stop");//start and pause camera capturing
        control.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    if (start.get() == true && capture.get().isOpened()) {
                        start.set(false);
                        capture.get().release();
                        //imwrite("color.jpg", colorimg);
                        control.setText("Start");
                    } else {
                        start.set(true);
                        capture.set(new VideoCapture());
                        capture.get().open(0);
                        control.setText("Stop");
                    }
                }
            }
        });
        mainframe.add(control, BorderLayout.CENTER);
        mainframe.pack();
        mainframe.setVisible(true);
        mainframe.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                if (capture.get().isOpened()) {
                    capture.get().release();
                }
                System.exit(0);
            }
        });

        CanvasFrame procframe = new CanvasFrame("Processed Frames - Emaraic");
        procframe.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        procframe.setCanvasSize(400, 400);
        procframe.setLocation(0, 0);
        CanvasFrame result = new CanvasFrame("Result - Emaraic");
        result.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        result.setCanvasSize(500, 500);
        result.setLocation(0, 440);

        while (true) {
            while (start.get() && capture.get().read(colorimg)) {
                if (mainframe.isVisible()) {

                    /*Convert to grayscale mode*/
                    Mat sourceGrey = new Mat(colorimg.size(), CV_8UC1);
                    cvtColor(colorimg, sourceGrey, COLOR_BGR2GRAY);
                    //imwrite("gray.jpg", new Mat(image)); // Save gray version of image

                    /*Apply Gaussian Filter*/
                    Mat blurimg = new Mat(colorimg.size(), CV_8UC1);
                    GaussianBlur(sourceGrey, blurimg, new Size(5, 5), 0);
                    //imwrite("blur.jpg", binimg);

                    /*Binarising Image*/
                    Mat binimg = new Mat(colorimg.size());
                    adaptiveThreshold(blurimg, binimg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 3);
                    //imwrite("binarise.jpg", binimg);

                    Rect r = getLargestRect(binimg);
                    Mat procimg = warpPrespectivePuzzle(binimg.clone());


                    /*opencv_imgproc.dilate(procimg, procimg, opencv_imgproc.getStructuringElement(opencv_imgproc.MORPH_RECT, new Size(5, 5)));
                    opencv_imgproc.erode(procimg, procimg, opencv_imgproc.getStructuringElement(opencv_imgproc.MORPH_RECT, new Size(3, 3)));
                    opencv_imgproc.morphologyEx(procimg, procimg, opencv_imgproc.MORPH_CLOSE, opencv_imgproc.getStructuringElement(opencv_imgproc.MORPH_RECT, new Size(2,2)),
                    new Point(0,0), 1, BORDER_CONSTANT, new Scalar());*/
                    Mat color = new Mat(colorimg);
                    if (isSudokuExist(binimg)) {
                        printCornerPoints(r, colorimg);
                        mainframe.showImage(converter.convert(colorimg));
                        bitwise_not(procimg, procimg);
                        Mat clonedf = new Mat(procimg.clone());
                        Mat canimg = new Mat(procimg.size());
                        Canny(procimg, canimg, 30, 90);
                        //imwrite("canny.jpg", canimg);

                        /* Apply Satndard Hough Line Transform */
                        Mat lines = new Mat();//vector stores the parameters (rho,theta) of the detected lines
                        //HoughLines(canimg, lines, 1, CV_PI / 180, 70,1,1, 0, CV_PI);
                        HoughLines(canimg, lines, 1, CV_PI / 180, 100);

                        FloatRawIndexer srcIndexer = lines.createIndexer();

                        /*Horizontal lines and one for vertical lines*/
                        List<org.deeplearning4j.clustering.cluster.Point> hpoints = new ArrayList<>();
                        List<org.deeplearning4j.clustering.cluster.Point> vpoints = new ArrayList<>();

                        for (int i = 0; i < srcIndexer.rows(); i++) {
                            float[] data = new float[2]; //data[0] is rho and data[1] is theta
                            srcIndexer.get(0, i, data);
                            double d[] = {data[0], data[1]};
                            if (Math.sin(data[1]) > 0.8) {//horizontal lines have a sin value equals 1, I just considered >.8 is horizontal line.
                                hpoints.add(new org.deeplearning4j.clustering.cluster.Point("hrho" + Math.sin(data[1]), "hrho", d));
                            } else if (Math.cos(data[1]) > 0.8) {//vertical lines have a cos value equals 1,
                                vpoints.add(new org.deeplearning4j.clustering.cluster.Point("vrho" + Math.cos(data[1]), "vrho", d));
                            }
                        }

                        /*Cluster vertical and horizontal lines into 10 lines for each using kmeans with 10 iterations*/
                        KMeansClustering kmeans = KMeansClustering.setup(10, 10, "euclidean");

                        log.info("Lines Number " + vpoints.size() + " " + hpoints.size());
                        if (vpoints.size() >= 10 && hpoints.size() >= 10) {
                            ClusterSet hcs = kmeans.applyTo(hpoints);
                            List<Cluster> hlines = hcs.getClusters();
                            Collections.sort(hlines, new LinesComparator());

                            ClusterSet vcs = kmeans.applyTo(vpoints);
                            List<Cluster> vlines = vcs.getClusters();
                            Collections.sort(vlines, new LinesComparator());
                            if (checkLines(vlines, hlines)) {
                                List<Point> points = getPoint(vlines, hlines);
                                if (points.size() != 100) {
                                    //break to get another image if number of points not equal 100
                                    break;
                                }

                                /*Print vertical lines, horizontal lines, and the intersection between them */
                                for (Point point : points) {
                                    circle(procimg, point, 10, new Scalar(0, 0, 0, 255), CV_FILLED, 8, 0);
                                }

                                vlines.addAll(hlines);//appen hlines to vlines to print them in one for loop
                                for (int i = 0; i < vlines.size(); i++) {
                                    Cluster get = vlines.get(i);
                                    double rho = get.getCenter().getArray().getDouble(0);
                                    double theta = get.getCenter().getArray().getDouble(1);
                                    double a = Math.cos(theta), b = Math.sin(theta);
                                    double x0 = a * rho, y0 = b * rho;
                                    CvPoint pt1 = cvPoint((int) Math.round(x0 + 1000 * (-b)), (int) Math.round(y0 + 1000 * (a))), pt2 = cvPoint((int) Math.round(x0 - 1000 * (-b)), (int) Math.round(y0 - 1000 * (a)));
                                    line(procimg, new Point(pt1.x(), pt1.y()),
                                            new Point(pt2.x(), pt2.y()), new Scalar(0, 0, 0, 0), 3, CV_AA, 0);

                                }

                                double puzzel[] = new double[81];
                                int j = 0;
                                //Form rectangles of 81 cells from the 100 intersection points
                                List<Rect> rects = new ArrayList<>();
                                for (int i = 0; i < points.size() - 11; i++) {
                                    int ri = i / 10;
                                    int ci = i % 10;
                                    if (ci != 9 && ri != 9) {
                                        Point get = points.get(i);
                                        Point get2 = points.get(i + 11);
                                        Rect r1 = new Rect(get, get2);
                                        //Rect r1 = new Rect(new Point(get.x()+5,get.y()+5), new Point(get2.x()-5,get2.y()-5));
                                        //imwrite("di\\points" + i + ".jpg", clonedf.apply(r1));
                                        if ((r1.x() + r1.width() <= clonedf.cols()) && (r1.y() + r1.height() <= clonedf.rows()) && r1.x() >= 0 && r1.y() >= 0) {
                                            Mat s = detectDigit(clonedf.apply(r1));
                                            rects.add(r1);
                                            //imwrite("di\\points" + i + ".jpg", s);
                                            if (s.cols() == 28 && s.rows() == 28) {
                                                puzzel[j] = recogniseDigit(s);
                                            } else {
                                                puzzel[j] = 0;
                                            }
                                            j++;
                                        }
                                    }
                                }
                                imwrite("procimg.jpg", procimg);
                                INDArray pd = Nd4j.create(puzzel);
                                INDArray puz = pd.reshape(new int[]{9, 9});
                                System.out.println(puz);
                                INDArray solvedpuz = puz.dup();
                                if (Sudoku.isValid(puzzel)) {
                                    //this code section is reponsible for if the solution of sudoku takes more than 5 second, break it.
                                    ExecutorService service = Executors.newSingleThreadExecutor();
                                    try {
                                        Future<Object> solver = (Future<Object>) service.submit(() -> {
                                            Sudoku.solve(0, 0, solvedpuz);
                                        });
                                        System.out.println(solver.get(5, TimeUnit.SECONDS));
                                    } catch (final TimeoutException e) {
                                        log.info("It takes a lot of time to solve, Going to break!!");
                                        /*break to get another image if sudoku solution takes more than 5 seconds
                                        sometime it takes along time for solving sudoku as a result of incorrect digit recognition.
                                        Mostely you face this when you rotate the puzzle */
                                        break;
                                    } catch (final Exception e) {
                                        log.error(e.getMessage());
                                    } finally {
                                        service.shutdown();
                                    }

                                    if (isContainsZero(solvedpuz)) {
                                        /*  putText(procimg, "CAN Not Solve It", new Point(0, procimg.cols() / 2),
                                        FONT_HERSHEY_COMPLEX, 1, new Scalar(0, 0, 0, 0), 3, 2, false);*/
                                        break; //break to get another image if solution is invalid
                                    } else {
                                        /*resimg = colorimg.apply(r);
                                        resize(resimg, resimg, new Size(600, 600));*/
                                        color = new Mat(procimg.size(), CV_8UC3);
                                        cvtColor(procimg, color, COLOR_GRAY2BGR);
                                        printResult(color, solvedpuz, puz, rects);
                                    }
                                } else {//break to get another image if sudoku is invalid
                                    break;
                                }
                                start.set(Boolean.FALSE);
                                capture.get().release();
                                control.setText("Try Again");
                            }//End if checkLines
                        }

                        procframe.showImage(converter.convert(procimg));
                        result.showImage(converter.convert(color));

                    } else {//End If sudoku puzzle exists
                        mainframe.showImage(converter.convert(colorimg));
                        procframe.showImage(converter.convert(procimg));
                        result.showImage(converter.convert(color));
                    }
                } else {//End if graabbed image equal null
                    System.out.println("Error!!!!");
                    System.exit(1);
                }
                try {
                    Thread.sleep(150);
                } catch (InterruptedException ex) {
                    log.error(ex.getMessage());
                }
            }//End While Start's Condition
            try {
                Thread.sleep(400);
            } catch (InterruptedException ex) {
                log.error(ex.getMessage());
            }
        }//End While True
    }
}
