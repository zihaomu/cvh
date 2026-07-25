#include "cvh.h"
#include "gtest/gtest.h"

#include <string>

using namespace cvh;

TEST(ErrorTest, cv_error_preserves_code_message_and_source)
{
    try
    {
        CV_Error(
            Error::StsNotImplemented,
            "feature is intentionally unavailable");
        FAIL() << "CV_Error must throw";
    }
    catch (const Exception& exception)
    {
        EXPECT_EQ(exception.code, Error::StsNotImplemented);
        EXPECT_EQ(
            exception.err,
            "feature is intentionally unavailable");
        EXPECT_FALSE(exception.file.empty());
        EXPECT_GT(exception.line, 0);
        EXPECT_NE(
            std::string(exception.what()).find(
                "feature is intentionally unavailable"),
            std::string::npos);
    }
}
