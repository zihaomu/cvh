#ifndef P2_BENCH_FUNCTION
#error "P2_BENCH_FUNCTION must be defined"
#endif
#ifndef P2_NAMESPACE
#error "P2_NAMESPACE must be defined"
#endif
#ifndef P2_MAT
#error "P2_MAT must be defined"
#endif
#ifndef P2_POINT_TYPE
#error "P2_POINT_TYPE must be defined"
#endif

double P2_BENCH_FUNCTION(Phase2OpId op,
                         int rows,
                         int cols,
                         int point_count,
                         int warmup,
                         int iters,
                         int repeats,
                         std::uint32_t seed)
{
    namespace api = P2_NAMESPACE;
    using Mat = P2_MAT;
    using Point = P2_POINT_TYPE;

    switch (op)
    {
        case Phase2OpId::RanduU8C3:
        case Phase2OpId::RandnU8C3:
        case Phase2OpId::RanduF32C3:
        case Phase2OpId::RandnF32C3:
        {
            const bool floating =
                op == Phase2OpId::RanduF32C3 ||
                op == Phase2OpId::RandnF32C3;
            const bool normal =
                op == Phase2OpId::RandnU8C3 ||
                op == Phase2OpId::RandnF32C3;
            Mat matrix = p2_make_mat(
                rows, cols, floating ? CV_32FC3 : CV_8UC3);
            return p2_measure_ms(
                [&]() {
                    if (normal)
                    {
                        api::randn(
                            matrix,
                            api::Scalar::all(64.0),
                            api::Scalar::all(12.0));
                    }
                    else
                    {
                        api::randu(
                            matrix,
                            api::Scalar::all(0.0),
                            api::Scalar::all(127.0));
                    }
                },
                [&]() { return p2_checksum(matrix); },
                warmup,
                iters,
                repeats);
        }
        case Phase2OpId::RanduU8C1Roi:
        {
            Mat storage = p2_make_mat(rows + 2, cols + 3, CV_8UC1);
            Mat roi = storage(
                api::Range(1, rows + 1),
                api::Range(2, cols + 2));
            return p2_measure_ms(
                [&]() {
                    api::randu(
                        roi, api::Scalar(0.0), api::Scalar(256.0));
                },
                [&]() { return p2_checksum(roi); },
                warmup,
                iters,
                repeats);
        }
        case Phase2OpId::TransformF32C3ToC4:
        case Phase2OpId::PerspectiveTransformF32C3:
        {
            Mat source = p2_make_mat(point_count, 1, CV_32FC3);
            p2_fill_f32(source, seed);
            Mat matrix = p2_make_mat(4, 4, CV_64FC1);
            matrix = 0.0f;
            for (int index = 0; index < 4; ++index)
            {
                matrix.template at<double>(index, index) = 1.0;
            }
            Mat destination;
            return p2_measure_ms(
                [&]() {
                    if (op == Phase2OpId::TransformF32C3ToC4)
                    {
                        api::transform(source, destination, matrix);
                    }
                    else
                    {
                        api::perspectiveTransform(
                            source, destination, matrix);
                    }
                },
                [&]() { return p2_checksum(destination); },
                warmup,
                iters,
                repeats);
        }
        case Phase2OpId::ConnectedComponents:
        case Phase2OpId::ConnectedComponentsWithStats:
        {
            Mat mask = p2_make_mat(rows, cols, CV_8UC1);
            p2_fill_region_mask(mask);
            Mat labels;
            Mat stats;
            Mat centroids;
            int component_count = 0;
            return p2_measure_ms(
                [&]() {
                    if (op == Phase2OpId::ConnectedComponents)
                    {
                        component_count = api::connectedComponents(
                            mask, labels, 8, CV_32S);
                    }
                    else
                    {
                        component_count =
                            api::connectedComponentsWithStats(
                                mask,
                                labels,
                                stats,
                                centroids,
                                8,
                                CV_32S);
                    }
                },
                [&]() {
                    return p2_checksum(
                               op == Phase2OpId::ConnectedComponents
                                   ? labels
                                   : stats) +
                        static_cast<double>(component_count);
                },
                warmup,
                iters,
                repeats);
        }
        case Phase2OpId::FindContours:
        {
            Mat mask = p2_make_mat(rows, cols, CV_8UC1);
            p2_fill_contour_mask(mask);
            std::vector<std::vector<Point>> contours;
            return p2_measure_ms(
                [&]() {
                    api::findContours(
                        mask,
                        contours,
                        api::RETR_LIST,
                        api::CHAIN_APPROX_SIMPLE);
                },
                [&]() { return p2_checksum_contours(contours); },
                warmup,
                iters,
                repeats);
        }
        case Phase2OpId::BoundingRect:
        case Phase2OpId::ContourArea:
        case Phase2OpId::ArcLength:
        case Phase2OpId::ApproxPolyDP:
        case Phase2OpId::ConvexHull:
        case Phase2OpId::IsContourConvex:
        case Phase2OpId::Moments:
        {
            const std::vector<Point> points =
                p2_make_shape_points(point_count);
            std::vector<Point> point_output;
            double scalar_output = 0.0;
            return p2_measure_ms(
                [&]() {
                    switch (op)
                    {
                        case Phase2OpId::BoundingRect:
                        {
                            const auto rectangle =
                                api::boundingRect(points);
                            scalar_output =
                                static_cast<double>(rectangle.x) +
                                static_cast<double>(rectangle.y) +
                                static_cast<double>(rectangle.width) +
                                static_cast<double>(rectangle.height);
                            break;
                        }
                        case Phase2OpId::ContourArea:
                            scalar_output = api::contourArea(points);
                            break;
                        case Phase2OpId::ArcLength:
                            scalar_output = api::arcLength(points, true);
                            break;
                        case Phase2OpId::ApproxPolyDP:
                            api::approxPolyDP(
                                points, point_output, 1.0, true);
                            break;
                        case Phase2OpId::ConvexHull:
                            api::convexHull(
                                points, point_output, false);
                            break;
                        case Phase2OpId::IsContourConvex:
                            scalar_output =
                                api::isContourConvex(points) ? 1.0 : 0.0;
                            break;
                        case Phase2OpId::Moments:
                            scalar_output = api::moments(points).m00;
                            break;
                        default:
                            break;
                    }
                },
                [&]() {
                    if (op == Phase2OpId::ApproxPolyDP ||
                        op == Phase2OpId::ConvexHull)
                    {
                        return p2_checksum_points(point_output);
                    }
                    return scalar_output;
                },
                warmup,
                iters,
                repeats);
        }
        case Phase2OpId::CalcHist:
        case Phase2OpId::CompareHistCorrel:
        case Phase2OpId::CompareHistChiSqr:
        case Phase2OpId::CompareHistIntersect:
        case Phase2OpId::CompareHistBhattacharyya:
        {
            Mat image = p2_make_mat(rows, cols, CV_8UC1);
            p2_fill_u8(image, seed);
            Mat histogram;
            p2_calc_hist(image, histogram);
            if (op == Phase2OpId::CalcHist)
            {
                return p2_measure_ms(
                    [&]() { p2_calc_hist(image, histogram); },
                    [&]() { return p2_checksum(histogram); },
                    warmup,
                    iters,
                    repeats);
            }

            Mat other = p2_make_mat(256, 1, CV_32FC1);
            other = 1.0f;
            int method = api::HISTCMP_CORREL;
            if (op == Phase2OpId::CompareHistChiSqr)
            {
                method = api::HISTCMP_CHISQR;
            }
            else if (op == Phase2OpId::CompareHistIntersect)
            {
                method = api::HISTCMP_INTERSECT;
            }
            else if (op == Phase2OpId::CompareHistBhattacharyya)
            {
                method = api::HISTCMP_BHATTACHARYYA;
            }
            double comparison = 0.0;
            return p2_measure_ms(
                [&]() {
                    comparison =
                        api::compareHist(histogram, other, method);
                },
                [&]() { return comparison; },
                warmup,
                iters,
                repeats);
        }
        case Phase2OpId::MatchTemplateSqDiff:
        case Phase2OpId::MatchTemplateSqDiffNormed:
        case Phase2OpId::MatchTemplateCCorr:
        case Phase2OpId::MatchTemplateCCorrNormed:
        {
            Mat image = p2_make_mat(rows, cols, CV_8UC1);
            p2_fill_u8(image, seed);
            constexpr int template_rows = 16;
            constexpr int template_cols = 16;
            Mat templ = image(
                api::Range(5, 5 + template_rows),
                api::Range(7, 7 + template_cols));
            Mat result;
            int method = api::TM_SQDIFF;
            if (op == Phase2OpId::MatchTemplateSqDiffNormed)
            {
                method = api::TM_SQDIFF_NORMED;
            }
            else if (op == Phase2OpId::MatchTemplateCCorr)
            {
                method = api::TM_CCORR;
            }
            else if (op == Phase2OpId::MatchTemplateCCorrNormed)
            {
                method = api::TM_CCORR_NORMED;
            }
            return p2_measure_ms(
                [&]() {
                    api::matchTemplate(image, templ, result, method);
                },
                [&]() { return p2_checksum(result); },
                warmup,
                iters,
                repeats);
        }
    }
    return 0.0;
}
